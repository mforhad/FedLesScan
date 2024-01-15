import abc
import logging
from functools import reduce
from typing import Iterator, Optional, List, Tuple

import numpy as np
import tensorflow as tf

from fedless.aggregator.exceptions import (
    InsufficientClientResults,
    UnknownCardinalityError,
)
from fedless.common.models import Parameters, ClientResult, TestMetrics
from fedless.common.models.aggregation_models import AggregationHyperParams
from fedless.common.persistence.client_daos import ClientResultDao

from fedless.common.serialization import deserialize_parameters

from fedless.aggregator.parameter_aggregator import ParameterAggregator

logger = logging.getLogger(__name__)


class StallAwareAggregator(ParameterAggregator):
    def __init__(self, current_round, aggregation_hyper_params: AggregationHyperParams):
        self.current_round = current_round
        self.tolerance = (
            aggregation_hyper_params.tolerance
            if aggregation_hyper_params is not None
            else 0
        )
        super().__init__()

    def _score_clients(self, client_result: List[dict]):
        scores = map(
            lambda client_dict: (client_dict["round_id"] + 1)
            / (self.current_round + 1),
            client_result,
        )
        return list(scores)

    def _aggregate(
        self,
        client_feats: List[dict],
        parameters: List[List[np.ndarray]],
        weights: List[float],
    ) -> List[np.ndarray]:
        # Partially from https://github.com/adap/flower/blob/
        # 570788c9a827611230bfa78f624a89a6630555fd/src/py/flwr/server/strategy/aggregate.py#L26
        # weights is a list of the number of elements for each client
        #
        num_examples_total = sum(weights)
        client_scores = self._score_clients(client_feats)
        # scale by the client scores
        weighted_weights = [
            [layer * num_examples * client_score for layer in weights]
            for weights, num_examples, client_score in zip(
                parameters, weights, client_scores
            )
        ]

        # noinspection PydanticTypeChecker,PyTypeChecker
        weights_prime: List[np.ndarray] = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime

    def select_aggregation_candidates(self, mongo_client, session_id, round_id):
        result_dao = ClientResultDao(mongo_client)
        # parameter_dao = ParameterDao(mongo_client)
        logger.debug(f"Establishing database connection")
        round_dicts, round_candidates = result_dao.load_results_for_session(
            session_id=session_id, round_id=round_id, tolerance=self.tolerance
        )
        if not round_candidates:
            raise InsufficientClientResults(
                f"Found no client results for session {session_id} and round {round_id}"
            )
        return round_dicts, round_candidates

    def aggregate(
        self,
        client_results: Iterator[ClientResult],
        client_feats: List[dict],
        default_cardinality: Optional[float] = None,
    ) -> Tuple[Parameters, Optional[List[TestMetrics]]]:
        client_parameters: List[List[np.ndarray]] = []
        client_cardinalities: List[int] = []
        client_metrics: List[TestMetrics] = []
        for client_result in client_results:
            params = deserialize_parameters(client_result.parameters)
            del client_result.parameters
            cardinality = client_result.cardinality

            # Check if cardinality is valid and handle accordingly
            if cardinality in [
                tf.data.UNKNOWN_CARDINALITY,
                tf.data.INFINITE_CARDINALITY,
            ]:
                if not default_cardinality:
                    raise UnknownCardinalityError(
                        f"Cardinality for client result invalid. "
                    )
                else:
                    cardinality = default_cardinality

            client_parameters.append(params)
            client_cardinalities.append(cardinality)
            if client_result.test_metrics:
                client_metrics.append(client_result.test_metrics)

        return (
            self._aggregate(client_feats, client_parameters, client_cardinalities),
            client_metrics or None,
        )


class StreamStallAwareAggregator(StallAwareAggregator):
    def __init__(
        self,
        current_round: int,
        aggregation_hyper_params: AggregationHyperParams,
        chunk_size: int = 25,
    ):
        super().__init__(current_round, aggregation_hyper_params)
        self.chunk_size = chunk_size

    def chunks(self, iterator: Iterator, n) -> Iterator[List]:
        buffer = []
        for el in iterator:
            if len(buffer) < n:
                buffer.append(el)
            if len(buffer) == n:
                yield buffer
                buffer = []
        else:
            if len(buffer) > 0:
                yield buffer

    def aggregate(
        self,
        client_results: Iterator[ClientResult],
        client_feats: List[dict],
        default_cardinality: Optional[float] = None,
    ) -> Tuple[Parameters, Optional[List[TestMetrics]]]:
        curr_global_params: Parameters = None
        curr_sum_weights = 0
        client_metrics: List[TestMetrics] = []
        for results_chunk in self.chunks(client_results, self.chunk_size):
            params_buffer, card_buffer = [], []
            for client_result in results_chunk:
                params = deserialize_parameters(client_result.parameters)
                del client_result.parameters
                cardinality = client_result.cardinality

                # Check if cardinality is valid and handle accordingly
                if cardinality in [
                    tf.data.UNKNOWN_CARDINALITY,
                    tf.data.INFINITE_CARDINALITY,
                ]:
                    if not default_cardinality:
                        raise UnknownCardinalityError(
                            f"Cardinality for client result invalid. "
                        )
                    else:
                        cardinality = default_cardinality

                params_buffer.append(params)
                card_buffer.append(cardinality)
                if client_result.test_metrics:
                    client_metrics.append(client_result.test_metrics)
            if curr_global_params is None:
                curr_global_params = self._aggregate(
                    client_feats, params_buffer, card_buffer
                )
            else:
                curr_global_params = self._aggregate(
                    client_feats,
                    [curr_global_params, *params_buffer],
                    [curr_sum_weights, *card_buffer],
                )
            curr_sum_weights += sum(card_buffer)

        return curr_global_params, client_metrics or None
