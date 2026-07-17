import sys
import subprocess
from pathlib import Path

def main():
    target_script = Path(__file__).parent / "output" / "main.py"
    
    if not target_script.exists():
        print(f"Error: Skrip utama tidak ditemukan di {target_script}")
        sys.exit(1)
        
    cmd = [sys.executable, str(target_script)] + sys.argv[1:]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
