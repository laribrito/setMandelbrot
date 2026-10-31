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
        f"g++ -O3 -ffast-math -march=native -std=c++17 -o ponto_a_ponto_v2 \"{os.path.join(src_dir, 'ponto_a_ponto_v2.cpp')}\"",
        f"g++ -O3 -ffast-math -fopenmp -std=c++17 -o paralelo \"{os.path.join(src_dir, 'paralelo.cpp')}\"",
        f"g++ -O3 -ffast-math -fopenmp -std=c++17 -o paralelo_collapse \"{os.path.join(src_dir, 'paralelo_collapse.cpp')}\"",
        f"g++ -O3 -o comparador \"{os.path.join(src_dir, 'comparador.cpp')}\"",
    ]

    import shutil
    has_mpicxx = shutil.which("mpicxx") is not None
    if has_mpicxx:
        cmds.append(f"mpicxx -O3 -ffast-math -std=c++17 -o mpi_estatico \"{os.path.join(src_dir, 'mpi_estatico.cpp')}\"")
        cmds.append(f"mpicxx -O3 -ffast-math -std=c++17 -o mpi_dinamico \"{os.path.join(src_dir, 'mpi_dinamico.cpp')}\"")
        cmds.append(f"mpicxx -O3 -ffast-math -std=c++17 -o mpi_dinamico_participativo \"{os.path.join(src_dir, 'mpi_dinamico_participativo.cpp')}\"")
        cmds.append(f"mpicxx -O3 -ffast-math -std=c++17 -o mpi_dinamico_meia_linha \"{os.path.join(src_dir, 'mpi_dinamico_meia_linha.cpp')}\"")
    
    # Adicionando o mingw64 ao PATH dinamicamente apenas no Windows para o g++ encontrar suas DLLs
    env = os.environ.copy()
    if sys.platform == "win32":
        env["PATH"] = "C:\\msys64\\mingw64\\bin;" + env.get("PATH", "")
    
    try:
        for cmd in cmds:
            print(f"Executando: {cmd}")
            subprocess.run(cmd, check=True, shell=True, env=env, cwd=root_dir)
        print("✅ Compilação concluída com sucesso.")
        if not has_mpicxx:
            print("\n⚠️  Aviso: 'mpicxx' não foi detectado no sistema.")
            print("   Para compilar os programas do Relatório 2 (MPI), instale o OpenMPI:")
            print("   - Linux (Ubuntu/Debian): sudo apt update && sudo apt install -y openmpi-bin libopenmpi-dev")
            print("   - Windows (MSYS2): pacman -S mingw-w64-x86_64-msmpi\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro durante a compilação: {e}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
