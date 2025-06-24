# act_test/run_vertex.py

import os
import subprocess
import sys

def main():
    # Find the script that was packaged alongside this module
    here   = os.path.dirname(__file__)
    script = os.path.join(here, "vertex_train.sh")
    if not os.path.exists(script):
        raise FileNotFoundError(f"{script} not found")

    # Rebuild args so that --opt=val becomes ["--opt","val"]
    forwarded_args = []
    for arg in sys.argv[1:]:
        if arg.startswith("--") and "=" in arg:
            opt, val = arg.split("=", 1)
            forwarded_args += [opt, val]
        else:
            forwarded_args.append(arg)

    # Run the bash script with the corrected args
    subprocess.run(["bash", script] + forwarded_args, check=True)

if __name__ == "__main__":
    main()
