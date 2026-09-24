import os
import subprocess
import sys

def main():
    # Encontra a raiz do repositório (onde está a pasta src)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir

    src_dir = os.path.join(root_dir, "src")
    if not os.path.isdir(src_dir):
        src_dir = root_dir

    cmds = [
        f"g++ -O3 -ffast-math -o sequencial \"{os.path.join(src_dir, 'sequencial.cpp')}\"",
        f"g++ -O3 -ffast-math -std=c++17 -o ponto_a_ponto \"{os.path.join(src_dir, 'ponto_a_ponto.cpp')}\"",
        f"g++ -O3 -ffast-math -fopenmp -std=c++17 -o paralelo \"{os.path.join(src_dir, 'paralelo.cpp')}\"",
        f"g++ -O3 -ffast-math -fopenmp -std=c++17 -o paralelo_collapse \"{os.path.join(src_dir, 'paralelo_collapse.cpp')}\"",
        f"g++ -O3 -o comparador \"{os.path.join(src_dir, 'comparador.cpp')}\"",
    ]
    
    # Adicionando o mingw64 ao PATH dinamicamente apenas no Windows para o g++ encontrar suas DLLs
    env = os.environ.copy()
    if sys.platform == "win32":
        env["PATH"] = "C:\\msys64\\mingw64\\bin;" + env.get("PATH", "")
    
    try:
        for cmd in cmds:
            print(f"Executando: {cmd}")
            subprocess.run(cmd, check=True, shell=True, env=env, cwd=root_dir)
        print("[OK] Compilacao de todas as versoes concluida com sucesso.")
    except subprocess.CalledProcessError as e:
        print(f"[ERRO] Erro durante a compilacao: {e}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
