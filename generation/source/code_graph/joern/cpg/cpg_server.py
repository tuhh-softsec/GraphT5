import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import requests
from code_graph.joern.joern_server import JoernServer
from code_graph.util import set_cpu_affinity

logger = logging.getLogger(__name__)


class CPGServer(JoernServer):
    def __init__(self, lang: Optional[str] = "cpp", verbose=True):
        super().__init__()
        self.port = None

        self.lang = lang

        assert self.lang in ["c", "cpp", "java"], "unsupported language"

        self.verbose = verbose

    def start(self):
        command = self.get_cpg_command()

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            preexec_fn=os.setsid,
        )

        set_cpu_affinity(self.process.pid)

        # start joern cpg frontend server
        for line in self.process.stdout:
            if self.verbose:
                print(line.strip())

            match = re.search(r"FrontendHTTPServer started on port (\d+)", line)
            if match:
                self.port = int(match.group(1))
                break

        if not self.port:
            error_msg = "Failed to start the server or capture the port"
            if hasattr(self.process, "stderr") and self.process.stderr:
                stderr_output = self.process.stderr.read()
                if stderr_output:
                    if isinstance(stderr_output, bytes):
                        stderr_output = stderr_output.decode("utf-8", errors="replace")
                    error_msg += f"\n{stderr_output}"
            raise Exception(error_msg)

        # time.sleep(2)

    def restart(self):
        self.stop()
        self.port = None
        self.start()

    def get_cpg_command(self):
        command = []

        if self.lang in ["c", "cpp"]:
            command.extend(
                [
                    "c2cpg.sh",
                    "-J-XX:ActiveProcessorCount=1",
                    ".",
                    "--server",
                    "--define",
                    "UNUSED_ATTR",
                    "--define",
                    "__user",
                    "--define",
                    "STATIC",
                    "--define",
                    "STACK_OF(type) struct stack_st_##type",
                ]
            )
        elif self.lang == "java":
            command.extend(
                [
                    "javasrc2cpg",
                    "-J-XX:ActiveProcessorCount=1",
                    ".",
                    "--server",
                    "--enable-file-content",
                ]
            )

        if self.verbose:
            command.append("--log-problems")

        return command

    def generate_cpg(self, input_path, output_path, params=None, max_retries=2):
        if not self.port:
            raise Exception("Server is not running.")

        response = None
        tries = max_retries

        while tries >= 0:
            try:
                url = f"http://localhost:{self.port}/run"
                data = [("input", input_path), ("output", output_path)]

                if params:
                    for key, values in params.items():
                        if not isinstance(values, list):
                            values = [values]
                        for value in values:
                            data.append((key, value))

                headers = {"Content-Type": "application/x-www-form-urlencoded"}

                response = requests.post(
                    url, data=urlencode(data), headers=headers, timeout=5
                )
                break
            except requests.exceptions.Timeout:
                if (tries := tries - 1) >= 0:
                    self.restart()

        if response is None:
            logger.error(f"Failed to generate CPG {input_path}, skipping ...")
            Path(output_path).touch()

            return output_path

        if response.status_code != 200:
            raise Exception(f"CPG generation failed: {response.text}")

        return response.text  # output directory

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
