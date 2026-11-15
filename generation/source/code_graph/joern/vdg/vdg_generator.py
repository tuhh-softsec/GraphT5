import json
import logging
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import orjson
import zmq
from code_graph.joern.joern_server import JoernServer
from code_graph.util import set_cpu_affinity

HELPERS_DIR = os.path.dirname(__file__)

JOERN_VDG_SCRIPT = os.path.join(HELPERS_DIR, "scripts/vdg.scala")
JOERN_SERIALIZER_SCRIPT = os.path.join(HELPERS_DIR, "scripts/graph.scala")
STATS_SCRIPT = os.path.join(HELPERS_DIR, "scripts/statistics.scala")

logger = logging.getLogger(__name__)


class VDGGenerator(JoernServer):
    def __init__(self, timeout=10, lang: Optional[str] = "cpp", verbose=False):
        super().__init__()

        self.scala_script = JOERN_VDG_SCRIPT
        self.serializer_script = JOERN_SERIALIZER_SCRIPT
        self.timeout = timeout
        self.can_timeout = self.timeout is not None
        self.lang = lang
        self.verbose = verbose

        self.zmq_context = None
        self.zmq_socket = None
        self.zmq_port = None

        assert self.lang in ["c", "cpp", "java"], "unsupported language"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def stop(self):
        super().stop()

        if self.zmq_socket:
            self.zmq_socket.setsockopt(zmq.LINGER, 0)
            self.zmq_socket.close()
            self.zmq_socket = None
        if self.zmq_context:
            self.zmq_context.destroy(linger=0)
            self.zmq_context = None

    def initialize_zmq(self):
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PAIR)
        self.zmq_socket.set_hwm(1)
        self.zmq_port = self.zmq_socket.bind_to_random_port("tcp://127.0.0.1")
        # self.zmq_socket.subscribe(b"")
        self.zmq_socket.RCVTIMEO = 1000
        return f"tcp://127.0.0.1:{self.zmq_port}"

    def generate_vdgs(self, source_files, cpg_files, stats_base_paths=None):
        if not cpg_files:
            logging.info("No files to process.")
            return

        assert not stats_base_paths or len(cpg_files) == len(stats_base_paths), (
            "cpg_files and stats_base_paths must have the same length!"
        )

        save_prefixes = [os.path.splitext(source)[0] for source in source_files]

        def pop_vdg():
            cpg_files.pop(0)
            source_files.pop(0)
            save_prefixes.pop(0)
            if stats_base_paths:
                stats_base_paths.pop(0)

        def has_vdg():
            return len(cpg_files) > 0

        try:
            remaining_retries = 3
            while has_vdg():
                with self.vdg_arguments(
                    cpg_files, save_prefixes, stats_base_paths
                ) as argument_file_path:
                    zmq_endpoint = self.initialize_zmq()

                    for vdg in self._generate_vdgs_impl(
                        zmq_endpoint,
                        argument_file_path,
                        cpg_files,
                        calc_stats=False,
                    ):
                        if vdg is not None:
                            if VDGGenerator.validate_vdg(vdg):
                                pop_vdg()
                                remaining_retries = 3
                                yield vdg
                            else:
                                if os.path.exists(vdg):
                                    try:
                                        os.unlink(vdg)
                                        logging.info(f"Removed invalid VDG file: {vdg}")
                                    except OSError as e:
                                        logging.error(
                                            f"Failed to remove invalid VDG file {vdg}: {e}"
                                        )
                                remaining_retries -= 1
                                if remaining_retries <= 0:
                                    remaining_retries = 3
                                    logger.error(f"Giving up on {vdg}, skipping.")
                                    pop_vdg()
                                    yield None
                                break
                        elif vdg is None:
                            remaining_retries -= 1
                            if remaining_retries <= 0:
                                remaining_retries = 3
                                logger.error(f"Giving up on {vdg}, skipping.")
                                pop_vdg()
                                yield None
                            break
                    else:
                        if not has_vdg():
                            return
                        remaining_retries -= 1
                        if remaining_retries <= 0:
                            remaining_retries = 3
                            logger.error(f"Giving up on {cpg_files[0]}, skipping.")
                            pop_vdg()
                            yield None

                    logger.warning("restarting Joern...")
        finally:
            pass

    def _generate_vdgs_impl(
        self, zmq_endpoint, argument_file, cpg_files, calc_stats=False
    ):
        command = [
            "joern",
            "-J-XX:ActiveProcessorCount=1",
            "-J-Xmx4G",
            "--script",
            self.scala_script,
            "--param",
            f"paramsFilePath={argument_file}",
            "--param",
            f"zmqEndpoint={zmq_endpoint}",
            "--param",
            f"lang={self.lang}",
            "--import",
            self.serializer_script,
        ]

        if calc_stats:
            command.append(f"--import {STATS_SCRIPT}")

        stderr_file = open(self._get_joern_log_file_path(), "a", buffering=8192)

        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
                preexec_fn=os.setsid,
            )

            set_cpu_affinity(self.process.pid)

            startup_complete = False
            process_timeout = self.timeout

            while True:
                if self.can_timeout and startup_complete and process_timeout <= 0:
                    logger.error(f"Joern process is no longer responding.. ")
                    break

                try:
                    message = self.zmq_socket.recv_string()
                    process_timeout = self.timeout

                    startup_complete = True

                    if message == "END":
                        logger.info("END!")
                        return True

                    try:
                        data = json.loads(message)
                        if data["type"] == "progress":
                            self.verbose and logger.info(
                                f"Progress: {data['current']}/{data['total']}"
                            )
                        elif data["type"] == "result":
                            assert str(cpg_files[0]) in data["cpg_file"], (
                                f"Received VDG {data['cpg_file']} but expected {cpg_files[0]} from Joern process!"
                            )
                            yield data["file"]
                        elif data["type"] == "nop":
                            assert str(cpg_files[0]) in data["cpg_file"], (
                                f"Received (already processed) VDG {data['cpg_file']} but expected {cpg_files[0]} from Joern process!"
                            )
                            yield data["file"]
                        elif data["type"] == "warning":
                            logger.warning(data["message"])
                        elif data["type"] == "debug":
                            self.verbose and logger.warning(data["message"])
                        elif data["type"] == "error":
                            logger.error(
                                f"Process {os.getpid()} terminated with error: {data['message']}"
                            )
                            yield None
                    except json.JSONDecodeError:
                        logger.warning(f"Received invalid JSON: {message}")

                except zmq.error.Again:
                    if self.process.poll() is not None:
                        break

                    if self.can_timeout and startup_complete:
                        process_timeout -= 1
                    continue

            logger.info(f"Detached from Joern process {os.getpid()}.")

            # at this point joern crashed
            return_code = self.process.wait(6)

            if return_code != 0:
                logger.warning(
                    f"Joern process {os.getpid()} crashed unexpectedly, check logs!"
                )
                raise subprocess.CalledProcessError(return_code, command)

        except Exception:
            logging.exception(f"Joern process stalled, cannot determine its state!")
            yield None
        finally:
            stderr_file.flush()
            stderr_file.close()

            self.stop()

    @contextmanager
    def vdg_arguments(
        self,
        cpg_files: List[str],
        save_prefixes: List[str],
        stats_base_paths: Optional[List[str]] = None,
    ) -> str:
        """Creates a temporary configuration file with the provided parameters.

        Args:
            cpg_files: List of CPG file paths
            save_prefixes: List of save prefix paths
            stats_base_paths: Optional list of statistics base paths

        Yields:
            str: Path to the temporary configuration file
        """
        os.makedirs(".sequelizer_tmp", exist_ok=True)

        base_name = f"joern_arguments_{os.getpid()}"
        file_path = Path(".sequelizer_tmp") / base_name
        counter = 1

        # Check if file exists and append _i if needed
        while file_path.exists():
            file_path = Path(".sequelizer_tmp") / f"{base_name}_{counter}"
            counter += 1

        try:
            resolved_cpgs = [str(Path(cpg).resolve()) for cpg in cpg_files]

            config_entries = {
                "filepaths": resolved_cpgs,
                "saveprefixes": save_prefixes,
            }

            if stats_base_paths:
                config_entries["statsbasepaths"] = stats_base_paths

            with open(file_path, "w") as temp_file:
                for key, values in config_entries.items():
                    temp_file.write(f'{key}="{",".join(values)}"\n')
                temp_file.flush()

            yield str(file_path)

        finally:
            pass
            # file_path.unlink(missing_ok=True)

    # @contextmanager
    # def vdg_arguments(
    #     self,
    #     cpg_files: List[str],
    #     save_prefixes: List[str],
    #     stats_base_paths: Optional[List[str]] = None,
    # ) -> str:
    #     """Creates a temporary configuration file with the provided parameters.
    #
    #     Args:
    #         cpg_files: List of CPG file paths
    #         save_prefixes: List of save prefix paths
    #         stats_base_paths: Optional list of statistics base paths
    #
    #     Yields:
    #         str: Path to the temporary configuration file
    #     """
    #     os.makedirs(".sequelizer_tmp", exist_ok=True)
    #
    #     try:
    #         with tempfile.NamedTemporaryFile(
    #             delete=False, mode="w", dir=".sequelizer_tmp"
    #         ) as temp_file:
    #             resolved_cpgs = [str(Path(cpg).resolve()) for cpg in cpg_files]
    #
    #             config_entries = {
    #                 "filepaths": resolved_cpgs,
    #                 "saveprefixes": save_prefixes,
    #             }
    #
    #             if stats_base_paths:
    #                 config_entries["statsbasepaths"] = stats_base_paths
    #
    #             for key, values in config_entries.items():
    #                 temp_file.write(f'{key}="{",".join(values)}"\n')
    #
    #             temp_file.flush()
    #             yield temp_file.name
    #     finally:
    #         Path(temp_file.name).unlink(missing_ok=True)

    def _get_joern_log_file_path(self):
        logs_dir = Path(".sequelizer_tmp/logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir / f"joern_log_{os.getpid()}.log"

    @staticmethod
    def validate_vdg(node_link_file):
        try:
            with open(node_link_file, "rb") as f:
                data = orjson.loads(f.read())
            if isinstance(data, dict) and "nodes" in data and "links" in data:
                if not data["nodes"]:
                    logging.warning("Found empty VDG")
                    return False
                ranges = data.get("graph", {}).get("vulnerableRanges", [])
                node_ids = {node["id"] for node in data["nodes"]}
                if ranges and not any(
                    bounds["start"] <= node_id <= bounds["end"]
                    for bounds in ranges
                    for node_id in node_ids
                ):
                    logging.warning("Vulnerable VDG range does not intersect a node")
                    return False
                return data
            else:
                logging.error(f"Invalid VDG structure in {node_link_file}")
                return False
        except orjson.JSONDecodeError:
            logging.error(f"Invalid JSON in {node_link_file}")
            return False
        except IOError:
            logging.error(f"Unable to read {node_link_file}")
            return False
