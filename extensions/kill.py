import os, re


def kill_nvidia_users():
    output = os.popen("fuser -v /dev/nvidia*").read()
    pids = re.findall(r"\d+", output)
    
    if pids:
        pid_string = " ".join(pids)
        print(f"Found PIDs: {pid_string}")

        os.system(f"kill -9 {pid_string}")
        print("Processes killed successfully")
    else:
        print("No PIDs found")


if __name__ == "__main__":
    kill_nvidia_users()
