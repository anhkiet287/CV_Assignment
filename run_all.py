
import subprocess, sys

def run(cmd):
    print(">>", cmd)
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        sys.exit(ret)

if __name__ == "__main__":
    run("make std_full")
