import json
import torch
import subprocess
import os
import time
import shlex
import signal # Import signal module
from datetime import datetime  # Add datetime module for timestamps
import argparse

def get_cuda_arch():
    arch = torch.cuda.get_device_capability()
    return f"{arch[0]}.{arch[1]}"

os.environ['TORCH_CUDA_ARCH_LIST'] = get_cuda_arch() # Set CUDA architecture list to avoid warnings

# --- Configuration ---
CONDA_BASE_PATH = os.getenv('CONDA_PREFIX')
if CONDA_BASE_PATH and 'envs' in CONDA_BASE_PATH:
    CONDA_BASE_PATH = CONDA_BASE_PATH.split('/envs/')[0]

# Get script directory as base directory for resolving relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Global variables ---
launched_processes_info = [] # Store information and log file handles of launched processes
original_sigint_handler = None

# --- Helper functions ---
def log_info(message):
    """Log output function with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def resolve_path(path, base_dir=None):
    """Resolve path, if relative path then resolve based on base_dir or script directory"""
    if os.path.isabs(path):
        return path
    base = base_dir if base_dir else SCRIPT_DIR
    return os.path.normpath(os.path.join(base, path))

# --- Functions ---

def get_conda_activate_command(conda_env_name):
    if not CONDA_BASE_PATH:
        raise ValueError("CONDA_BASE_PATH not set, cannot locate Conda.")
    env_python_path = os.path.join(CONDA_BASE_PATH, 'envs', conda_env_name, 'bin', 'python')
    if os.name == 'nt': # Windows specific path
        env_python_path = os.path.join(CONDA_BASE_PATH, 'envs', conda_env_name, 'python.exe')

    if not os.path.exists(env_python_path):
        raise FileNotFoundError(f"Python interpreter not found in Conda environment '{conda_env_name}': {env_python_path}")
    return env_python_path

def launch_tool(tool_config):
    log_info(f"--- Starting tool: {tool_config['name']} ---")
    log_file_obj_stdout = None # Used to save opened log file object

    try:
        env_python = get_conda_activate_command(tool_config['conda_env'])
    except (ValueError, FileNotFoundError) as e:
        log_info(f"Error: {e}")
        log_info(f"Cannot start tool {tool_config['name']}.")
        return None, None

    uvicorn_command = [
        env_python,
        "-m", "uvicorn",
        tool_config['app_module'],
        "--host", "0.0.0.0",
        "--port", str(tool_config['port'])
    ]
    if tool_config.get('workers'):
        uvicorn_command.extend(["--workers", str(tool_config['workers'])])
    if tool_config.get('reload'):
        uvicorn_command.append("--reload")

    current_env = os.environ.copy()
    if tool_config.get('gpu_ids'):
        current_env["CUDA_VISIBLE_DEVICES"] = tool_config['gpu_ids']
        log_info(f"  Set CUDA_VISIBLE_DEVICES={tool_config['gpu_ids']}")
    else:
        if "CUDA_VISIBLE_DEVICES" in current_env:
            del current_env["CUDA_VISIBLE_DEVICES"]
        log_info(f"  No specific GPU specified (CUDA_VISIBLE_DEVICES not set or cleared)")

    # Inject extra environment variables defined in config's "env" field
    extra_env = tool_config.get('env', {})
    if extra_env:
        for k, v in extra_env.items():
            current_env[k] = str(v)
            log_info(f"  Set env {k}={v}")

    # Handle relative paths in working directory
    working_directory = tool_config['path']
    if not os.path.isabs(working_directory):
        # Resolve relative path to absolute path
        working_directory = resolve_path(working_directory)
        log_info(f"  Relative working directory resolved to: {working_directory}")
    
    # Ensure working directory exists
    if not os.path.exists(working_directory):
        log_info(f"  Warning: Working directory '{working_directory}' does not exist!")
    elif not os.path.isdir(working_directory):
        log_info(f"  Warning: Specified working directory '{working_directory}' is not a directory!")
    
    log_info(f"  Working directory: {working_directory}")
    log_info(f"  Command: {' '.join(uvicorn_command)}")

    # Handle log file path
    logfile_path = tool_config.get('logfile')
    stdout_arg_for_popen = subprocess.PIPE
    stderr_arg_for_popen = subprocess.PIPE

    if logfile_path:
        # If log path is relative, convert it to absolute path relative to working directory
        if not os.path.isabs(logfile_path):
            logfile_path = resolve_path(logfile_path, working_directory)
            log_info(f"  Relative log file path resolved to: {logfile_path}")
        
        try:
            # Ensure log file directory exists
            log_dir = os.path.dirname(logfile_path)
            os.makedirs(log_dir, exist_ok=True)
            log_info(f"  Ensure log directory exists: {log_dir}")
            
            with open(logfile_path, 'w', encoding='utf-8') as log_file_obj:
                pass
            
            log_file_obj_stdout = open(logfile_path, 'a', encoding='utf-8')
            stdout_arg_for_popen = log_file_obj_stdout
            stderr_arg_for_popen = subprocess.STDOUT # Redirect stderr to stdout log file
            log_info(f"  Logs will be output to: {logfile_path}")
        except Exception as e:
            log_info(f"  Warning: Cannot open log file {logfile_path}: {e}. Output will go to pipe.")
            if log_file_obj_stdout: # If file was opened but error occurred later
                log_file_obj_stdout.close()
            log_file_obj_stdout = None # Reset to None
            stdout_arg_for_popen = subprocess.PIPE
            stderr_arg_for_popen = subprocess.PIPE


    # On POSIX systems, use setsid to make child process session leader，
    # so entire process group can be killed with os.killpg.
    # Note this also means it won't directly receive signals from parent's controlling terminal (like Ctrl+C).
    # We'll handle this explicitly in parent process.
    preexec_fn_arg = None
    if os.name == 'posix':
        preexec_fn_arg = os.setsid

    # Windows specific creation flags for better signal handling if needed，
    # but start_new_session equivalent is usually os.setsid for Popen on POSIX.
    # For Windows, CREATE_NEW_PROCESS_GROUP might be useful if sending CTRL_BREAK_EVENT.
    creationflags = 0
    if os.name == 'nt':
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP


    try:
        process = subprocess.Popen(
            uvicorn_command,
            cwd=working_directory,
            env=current_env,
            stdout=stdout_arg_for_popen,
            stderr=stderr_arg_for_popen,
            preexec_fn=preexec_fn_arg, # Called in child process before exec (POSIX only)
            creationflags=creationflags # Windows only
        )
        log_info(f"  Tool {tool_config['name']} started, PID: {process.pid}, Port: {tool_config['port']}")
        return process, log_file_obj_stdout
    except FileNotFoundError:
        log_info(f"  Error: Command or path not found. Please ensure Conda environment '{tool_config['conda_env']}' is correct.")
        if log_file_obj_stdout: log_file_obj_stdout.close()
        return None, None
    except Exception as e:
        log_info(f"  Error starting tool {tool_config['name']}: {e}")
        if log_file_obj_stdout: log_file_obj_stdout.close()
        return None, None

def cleanup_processes():
    global launched_processes_info
    if not launched_processes_info:
        print("No running child processes to stop.")
        return

    print(f"\n--- Starting cleanup procedure, preparing to stop {len(launched_processes_info)} tools... ---")

    for p_info in reversed(launched_processes_info): # Stop from back to front
        process = p_info["process"]
        name = p_info["name"]
        log_file_stdout = p_info.get("log_file_stdout") # Get log file object

        if process.poll() is not None: # Process already ended
            print(f"Tool {name} (PID: {process.pid if hasattr(process, 'pid') else 'N/A'}) already ended before attempting to stop.")
            if log_file_stdout:
                try: log_file_stdout.close()
                except Exception as e_close: print(f"  Error closing log file for {name}: {e_close}")
            continue

        pid = process.pid # After process might end, pid might not be available
        print(f"Stopping tool {name} (PID: {pid})...")

        try:
            if os.name == 'posix':
                print(f"  (POSIX) Sending SIGINT to process group {os.getpgid(pid)}...")
                os.killpg(os.getpgid(pid), signal.SIGINT) # Send SIGINT to entire process group
            elif os.name == 'nt':
                print(f"  (Windows) Sending CTRL_BREAK_EVENT to process group {pid}...")
                # process.send_signal(signal.CTRL_C_EVENT) # Might not be enough
                # For CREATE_NEW_PROCESS_GROUP, CTRL_BREAK_EVENT more likely affects entire group
                process.send_signal(signal.CTRL_BREAK_EVENT)

            process.wait(timeout=10) # Wait for graceful shutdown
            print(f"  Tool {name} (PID: {pid}) successfully stopped (graceful shutdown).")

        except subprocess.TimeoutExpired:
            print(f"  Tool {name} (PID: {pid}) did not stop gracefully within 10 seconds. Trying terminate()...")
            process.terminate() # SIGTERM (POSIX) / TerminateProcess (Windows)
            try:
                process.wait(timeout=5)
                print(f"  Tool {name} (PID: {pid}) stopped via terminate().")
            except subprocess.TimeoutExpired:
                print(f"  Tool {name} (PID: {pid}) did not stop via terminate() within 5 seconds. Force kill()...")
                process.kill() # SIGKILL (POSIX) / TerminateProcess (Windows, try again)
                try:
                    process.wait(timeout=2)
                    print(f"  Tool {name} (PID: {pid}) force stopped via kill().")
                except Exception as e_kill_wait:
                    print(f"  Warning: Tool {name} (PID: {pid}) kill() wait error/not ended: {e_kill_wait}")
            except Exception as e_term_wait:
                print(f"  Tool {name} (PID: {pid}) terminate() wait error: {e_term_wait}")
        except Exception as e_signal: # Other errors related to signal sending or waiting
            print(f"  Error sending initial shutdown signal to {name} (PID: {pid}): {e_signal}")
            if process.poll() is None: # If still running
                print(f"  Trying direct force terminate {name} (PID: {pid})...")
                try:
                    process.kill()
                    process.wait(timeout=2)
                    print(f"  Tool {name} (PID: {pid}) directly force terminated.")
                except Exception as e_force_kill:
                    print(f"  Direct force terminate {name} (PID: {pid}) also failed: {e_force_kill}")
        finally:
            if log_file_stdout: # Ensure log file handle is closed
                try:
                    print(f"  Closing log file for {name}...")
                    log_file_stdout.close()
                except Exception as e_close:
                    print(f"  Error closing log file for {name}: {e_close}")
    
    launched_processes_info.clear() # Clear list
    print("--- All tool shutdown processing complete ---")

def handle_sigint(sig, frame):
    global original_sigint_handler
    print("\n--- Ctrl+C (SIGINT) detected! Starting shutdown of all tools... ---")
    # Restore original SIGINT handler to prevent issues if Ctrl+C pressed again during cleanup
    if original_sigint_handler:
        signal.signal(signal.SIGINT, original_sigint_handler)
    # Don't directly call cleanup_processes here, let main loop's finally block handle it
    # This ensures cleanup is executed even if exiting for other reasons (e.g., other exceptions)
    # Here we raise KeyboardInterrupt, which will be caught by main try...except
    raise KeyboardInterrupt


# --- Main logic ---
if __name__ == "__main__":
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handle_sigint)
    
    parser = argparse.ArgumentParser(description="Start tools defined in configuration file")
    parser.add_argument('--config', type=str, required=True, help='Configuration file path')
    args = parser.parse_args()
    CONFIG_FILE = args.config

    try:
        with open(CONFIG_FILE, 'r') as f:
            all_tools_config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{CONFIG_FILE}' not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Configuration file '{CONFIG_FILE}' format error.")
        exit(1)

    if not CONDA_BASE_PATH:
        print("Warning: CONDA_PREFIX environment variable not found or does not point to Conda base directory.")
        try:
            result = subprocess.run(['conda', 'info', '--base'], capture_output=True, text=True, check=True, timeout=5)
            CONDA_BASE_PATH = result.stdout.strip()
            print(f"Auto-detected Conda base path: {CONDA_BASE_PATH}")
        except Exception as e:
            print(f"Cannot auto-detect Conda base path: {e}. Please manually set CONDA_BASE_PATH in script or ensure conda is in PATH.")
            exit(1)

    try:
        for config in all_tools_config:
            process, log_stdout_fobj = launch_tool(config)
            if process:
                launched_processes_info.append({
                    "name": config["name"],
                    "process": process,
                    "port": config["port"],
                    "log_file_stdout": log_stdout_fobj, # Store file object
                    "config": config # Store original config for restart operations
                })
            time.sleep(1)

        print("\n--- All configured tools have been attempted to start ---")
        print("Main script is running... Press Ctrl+C to stop all tools.")

        while True:
            still_running_count = 0
            for i in range(len(launched_processes_info) -1, -1, -1): # Reverse traverse for removal
                p_info = launched_processes_info[i]
                process = p_info["process"]
                if process.poll() is None: # Process still running
                    still_running_count +=1
                else: # Process stopped
                    print(f"\nWarning: Tool {p_info['name']} (PID: {process.pid if hasattr(process,'pid') else 'N/A'}, Port: {p_info['port']}) stopped by itself.")
                    print(f"  Return code: {process.returncode}")
                    # Optional: log or try restart
                    if p_info["log_file_stdout"]:
                        try: p_info["log_file_stdout"].close()
                        except Exception: pass # Ignore close errors since process ended
                    launched_processes_info.pop(i) # Remove from list


            if still_running_count == 0 and any("process" in t for t in all_tools_config): # List not empty but all processes stopped
                print("All previously started tools have stopped. Script exiting.")
                break
            if not launched_processes_info and any("process" in t for t in all_tools_config): # If list became empty
                print("All tools have stopped. Script exiting.") # (This condition might duplicate above, adjust logic accordingly)
                break

            time.sleep(5)

    except KeyboardInterrupt: # Raised by handle_sigint, or directly by Ctrl+C (if signal handler not set in time)
        print("\n--- Main script caught KeyboardInterrupt ---")
    finally:
        print("\n--- Main script entering finally cleanup block ---")
        cleanup_processes()
        # Ensure original SIGINT handler is restored in case script doesn't exit normally
        if original_sigint_handler and signal.getsignal(signal.SIGINT) != original_sigint_handler:
             signal.signal(signal.SIGINT, original_sigint_handler)