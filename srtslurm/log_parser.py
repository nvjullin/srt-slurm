"""
Node analysis service for parsing .err log files

All parsing logic encapsulated in the NodeAnalyzer class.
"""

import logging
import os
import re

# Configure logging
logger = logging.getLogger(__name__)


class NodeAnalyzer:
    """Service for analyzing node-level metrics from log files.

    Parses .err files to extract batch metrics, memory usage, and configuration.
    All parsing logic is encapsulated as methods.
    """

    def parse_run_logs(self, run_path: str) -> list:
        """Parse all node log files in a run directory.

        Args:
            run_path: Path to the run directory containing .err files

        Returns:
            List of NodeMetrics objects, one per node
        """

        nodes = []

        if not os.path.exists(run_path):
            logger.error(f"Run path does not exist: {run_path}")
            return nodes

        total_err_files = 0
        parsed_successfully = 0

        for file in os.listdir(run_path):
            if file.endswith(".err") and ("prefill" in file or "decode" in file):
                total_err_files += 1
                filepath = os.path.join(run_path, file)
                node = self.parse_single_log(filepath)
                if node:
                    nodes.append(node)
                    parsed_successfully += 1

        logger.info(
            f"Parsed {parsed_successfully}/{total_err_files} prefill/decode .err files from {run_path}"
        )

        if total_err_files == 0:
            logger.warning(f"No prefill/decode .err files found in {run_path}")

        return nodes

    def parse_single_log(self, filepath: str):
        """Parse a single node log file.

        Args:
            filepath: Path to the .err log file

        Returns:
            NodeMetrics object or None if parsing failed
        """
        from .models import BatchMetrics, MemoryMetrics, NodeMetrics

        node_info = self._extract_node_info_from_filename(filepath)
        if not node_info:
            logger.warning(
                f"Could not extract node info from filename: {filepath}. "
                f"Expected format: <node>_<service>_<id>.err"
            )
            return None

        batches = []
        memory_snapshots = []
        config = {}

        try:
            with open(filepath) as f:
                for line in f:
                    # Parse prefill batch metrics
                    batch_metrics = self._parse_prefill_batch_line(line)
                    if batch_metrics:
                        batches.append(
                            BatchMetrics(
                                timestamp=batch_metrics["timestamp"],
                                dp=batch_metrics["dp"],
                                tp=batch_metrics["tp"],
                                ep=batch_metrics["ep"],
                                batch_type=batch_metrics["type"],
                                new_seq=batch_metrics.get("new_seq"),
                                new_token=batch_metrics.get("new_token"),
                                cached_token=batch_metrics.get("cached_token"),
                                token_usage=batch_metrics.get("token_usage"),
                                running_req=batch_metrics.get("running_req"),
                                queue_req=batch_metrics.get("queue_req"),
                                prealloc_req=batch_metrics.get("prealloc_req"),
                                inflight_req=batch_metrics.get("inflight_req"),
                                input_throughput=batch_metrics.get("input_throughput"),
                            )
                        )

                    # Parse decode batch metrics
                    decode_metrics = self._parse_decode_batch_line(line)
                    if decode_metrics:
                        batches.append(
                            BatchMetrics(
                                timestamp=decode_metrics["timestamp"],
                                dp=decode_metrics["dp"],
                                tp=decode_metrics["tp"],
                                ep=decode_metrics["ep"],
                                batch_type=decode_metrics["type"],
                                running_req=decode_metrics.get("running_req"),
                                queue_req=decode_metrics.get("queue_req"),
                                prealloc_req=decode_metrics.get("prealloc_req"),
                                transfer_req=decode_metrics.get("transfer_req"),
                                token_usage=decode_metrics.get("token_usage"),
                                preallocated_usage=decode_metrics.get("preallocated_usage"),
                                num_tokens=decode_metrics.get("num_tokens"),
                                gen_throughput=decode_metrics.get("gen_throughput"),
                            )
                        )

                    # Parse memory metrics
                    mem_metrics = self._parse_memory_line(line)
                    if mem_metrics:
                        memory_snapshots.append(
                            MemoryMetrics(
                                timestamp=mem_metrics["timestamp"],
                                dp=mem_metrics["dp"],
                                tp=mem_metrics["tp"],
                                ep=mem_metrics["ep"],
                                metric_type=mem_metrics["type"],
                                avail_mem_gb=mem_metrics.get("avail_mem_gb"),
                                mem_usage_gb=mem_metrics.get("mem_usage_gb"),
                                kv_cache_gb=mem_metrics.get("kv_cache_gb"),
                                kv_tokens=mem_metrics.get("kv_tokens"),
                            )
                        )

                    # Extract TP/DP/EP configuration from command line
                    if "--tp-size" in line:
                        tp_match = re.search(r"--tp-size\s+(\d+)", line)
                        dp_match = re.search(r"--dp-size\s+(\d+)", line)
                        ep_match = re.search(r"--ep-size\s+(\d+)", line)

                        if tp_match:
                            config["tp_size"] = int(tp_match.group(1))
                        if dp_match:
                            config["dp_size"] = int(dp_match.group(1))
                        if ep_match:
                            config["ep_size"] = int(ep_match.group(1))

        except Exception as e:
            logger.error(f"Error parsing {filepath}: {e}")
            return None

        # Validation: Log if we found no metrics
        total_metrics = len(batches) + len(memory_snapshots)

        if total_metrics == 0:
            logger.warning(
                f"Parsed {filepath} but found no metrics. "
                f"Expected to find lines with DP/TP/EP tags. "
                f"Log format may have changed."
            )

        logger.debug(
            f"Parsed {filepath}: {len(batches)} batches, "
            f"{len(memory_snapshots)} memory snapshots"
        )

        return NodeMetrics(
            node_info=node_info,
            batches=batches,
            memory_snapshots=memory_snapshots,
            config=config,
        )

    def get_prefill_nodes(self, nodes: list):
        """Filter for prefill nodes only.

        Args:
            nodes: List of NodeMetrics objects

        Returns:
            Filtered list containing only prefill nodes
        """
        return [n for n in nodes if n.is_prefill]

    def get_decode_nodes(self, nodes: list):
        """Filter for decode nodes only.

        Args:
            nodes: List of NodeMetrics objects

        Returns:
            Filtered list containing only decode nodes
        """
        return [n for n in nodes if n.is_decode]

    def get_node_count(self, run_path: str) -> tuple[int, int]:
        """Get count of prefill and decode nodes in a run.

        Args:
            run_path: Path to the run directory

        Returns:
            Tuple of (prefill_count, decode_count)
        """
        nodes = self.parse_run_logs(run_path)

        prefill_count = sum(1 for n in nodes if n.is_prefill)
        decode_count = sum(1 for n in nodes if n.is_decode)

        return (prefill_count, decode_count)

    def has_batch_metrics(self, nodes: list) -> bool:
        """Check if any node has batch-level metrics.

        Useful for detecting if decode nodes are logging batch metrics.

        Args:
            nodes: List of NodeMetrics objects

        Returns:
            True if any node has batch data
        """
        return any(len(n.batches) > 0 for n in nodes)

    # Private helper methods

    def _parse_dp_tp_ep_tag(
        self, line: str
    ) -> tuple[int | None, int | None, int | None, str | None]:
        """Extract DP, TP, EP indices and timestamp from log line.

        Supports two formats:
        - Full: [2025-11-04 05:31:43 DP0 TP0 EP0]
        - Simple: [2025-11-04 07:05:55 TP0] (defaults DP=0, EP=0)

        Args:
            line: Log line to parse

        Returns:
            (dp, tp, ep, timestamp) or (None, None, None, None) if pattern not found
        """
        # Try full format first: DP0 TP0 EP0
        match = re.search(
            r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) DP(\d+) TP(\d+) EP(\d+)\]", line
        )
        if match:
            timestamp, dp, tp, ep = match.groups()
            return int(dp), int(tp), int(ep), timestamp

        # Try simple format: TP0 only (1P4D style)
        match = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) TP(\d+)\]", line)
        if match:
            timestamp, tp = match.groups()
            return 0, int(tp), 0, timestamp  # Default DP=0, EP=0

        return None, None, None, None

    def _parse_prefill_batch_line(self, line: str) -> dict | None:
        """Parse prefill batch log line for metrics.

        Example line:
        [2025-11-04 05:31:43 DP0 TP0 EP0] Prefill batch, #new-seq: 18, #new-token: 16384,
        #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 0,
        #prealloc-req: 0, #inflight-req: 0, input throughput (token/s): 0.00,
        """
        dp, tp, ep, timestamp = self._parse_dp_tp_ep_tag(line)
        if dp is None or "Prefill batch" not in line:
            return None

        metrics = {"timestamp": timestamp, "dp": dp, "tp": tp, "ep": ep, "type": "prefill"}

        # Extract metrics using regex
        patterns = {
            "new_seq": r"#new-seq:\s*(\d+)",
            "new_token": r"#new-token:\s*(\d+)",
            "cached_token": r"#cached-token:\s*(\d+)",
            "token_usage": r"token usage:\s*([\d.]+)",
            "running_req": r"#running-req:\s*(\d+)",
            "queue_req": r"#queue-req:\s*(\d+)",
            "prealloc_req": r"#prealloc-req:\s*(\d+)",
            "inflight_req": r"#inflight-req:\s*(\d+)",
            "input_throughput": r"input throughput \(token/s\):\s*([\d.]+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                value = match.group(1)
                metrics[key] = float(value) if "." in value else int(value)

        return metrics

    def _parse_decode_batch_line(self, line: str) -> dict | None:
        """Parse decode batch log line for metrics.

        Example line:
        [2025-11-04 05:32:32 DP31 TP31 EP31] Decode batch, #running-req: 7, #token: 7040,
        token usage: 0.00, pre-allocated usage: 0.00, #prealloc-req: 0, #transfer-req: 0,
        #retracted-req: 0, cuda graph: True, gen throughput (token/s): 6.73, #queue-req: 0,
        """
        dp, tp, ep, timestamp = self._parse_dp_tp_ep_tag(line)
        if dp is None or "Decode batch" not in line:
            return None

        metrics = {"timestamp": timestamp, "dp": dp, "tp": tp, "ep": ep, "type": "decode"}

        # Extract metrics using regex
        patterns = {
            "running_req": r"#running-req:\s*(\d+)",
            "num_tokens": r"#token:\s*(\d+)",
            "token_usage": r"token usage:\s*([\d.]+)",
            "preallocated_usage": r"pre-allocated usage:\s*([\d.]+)",
            "prealloc_req": r"#prealloc-req:\s*(\d+)",
            "transfer_req": r"#transfer-req:\s*(\d+)",
            "queue_req": r"#queue-req:\s*(\d+)",
            "gen_throughput": r"gen throughput \(token/s\):\s*([\d.]+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                value = match.group(1)
                metrics[key] = float(value) if "." in value else int(value)

        return metrics

    def _parse_memory_line(self, line: str) -> dict | None:
        """Parse memory-related log lines.

        Examples:
        [2025-11-04 05:27:13 DP0 TP0 EP0] Load weight end. type=DeepseekV3ForCausalLM,
        dtype=torch.bfloat16, avail mem=75.11 GB, mem usage=107.07 GB.

        [2025-11-04 05:27:13 DP0 TP0 EP0] KV Cache is allocated. #tokens: 524288, KV size: 17.16 GB
        """
        dp, tp, ep, timestamp = self._parse_dp_tp_ep_tag(line)
        if dp is None:
            return None

        metrics = {
            "timestamp": timestamp,
            "dp": dp,
            "tp": tp,
            "ep": ep,
        }

        # Parse available memory
        avail_match = re.search(r"avail mem=([\d.]+)\s*GB", line)
        if avail_match:
            metrics["avail_mem_gb"] = float(avail_match.group(1))
            metrics["type"] = "memory"

        # Parse memory usage
        usage_match = re.search(r"mem usage=([\d.]+)\s*GB", line)
        if usage_match:
            metrics["mem_usage_gb"] = float(usage_match.group(1))
            metrics["type"] = "memory"

        # Parse KV cache size
        kv_match = re.search(r"KV size:\s*([\d.]+)\s*GB", line)
        if kv_match:
            metrics["kv_cache_gb"] = float(kv_match.group(1))
            metrics["type"] = "kv_cache"

        # Parse token count for KV cache
        token_match = re.search(r"#tokens:\s*(\d+)", line)
        if token_match:
            metrics["kv_tokens"] = int(token_match.group(1))

        return metrics if "type" in metrics else None

    def _extract_node_info_from_filename(self, filename: str) -> dict | None:
        """Extract node name and worker info from filename.

        Example: watchtower-navy-cn01_prefill_w0.err
        Returns: {'node': 'watchtower-navy-cn01', 'worker_type': 'prefill', 'worker_id': 'w0'}
        """
        match = re.match(
            r"([^_]+)_(prefill|decode|frontend)_([^.]+)\.err", os.path.basename(filename)
        )
        if match:
            return {
                "node": match.group(1),
                "worker_type": match.group(2),
                "worker_id": match.group(3),
            }
        return None


# Standalone helper function for visualizations
def get_node_label(node_data: dict) -> str:
    """Generate a display label for a node with its configuration.

    Example: "[3320] cn01-prefill-w0 (DP0-3, TP8, EP8)"
    """
    node_info = node_data["node_info"]
    config = node_data["config"]
    run_id = node_data.get("run_id", "")

    node_name = node_info["node"].replace("watchtower-navy-", "").replace("watchtower-aqua-", "")
    worker = f"{node_info['worker_type']}-{node_info['worker_id']}"

    # Get DP range if we have batch data
    dp_indices = set()
    for batch in node_data.get("prefill_batches", []):
        if "dp" in batch:
            dp_indices.add(batch["dp"])

    if dp_indices:
        dp_min, dp_max = min(dp_indices), max(dp_indices)
        if dp_min == dp_max:
            dp_str = f"DP{dp_min}"
        else:
            dp_str = f"DP{dp_min}-{dp_max}"
    else:
        dp_str = "DP?"

    tp_str = f"TP{config.get('tp_size', '?')}"
    ep_str = f"EP{config.get('ep_size', '?')}"

    # Include run_id prefix if available (for multi-run comparisons)
    if run_id:
        # Extract just the job number from directory name like "3320_1P_4D_20251104_231843"
        run_prefix = run_id.split("_")[0] if "_" in run_id else run_id
        return f"[{run_prefix}] {node_name}-{worker} ({dp_str}, {tp_str}, {ep_str})"
    else:
        return f"{node_name}-{worker} ({dp_str}, {tp_str}, {ep_str})"
