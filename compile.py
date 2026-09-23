import os
import subprocess
import sys

def main():
    command1 = "g++ -O3 -ffast-math -o sequencial sequencial.cpp"
    command2 = "g++ -O3 -ffast-math -std=c++17 -o ponto_a_ponto ponto_a_ponto.cpp"
    command3 = "g++ -O3 -ffast-math -fopenmp -std=c++17 -o paralelo paralelo.cpp"
    command4 = "g++ -O3 -o comparador comparador.cpp"
    
    # Adicionando o mingw64 ao PATH dinamicamente apenas no Windows para o g++ encontrar suas DLLs
    env = os.environ.copy()
    if sys.platform == "win32":
        env["PATH"] = "C:\\msys64\\mingw64\\bin;" + env.get("PATH", "")
    
    try:
        print(f"Executando: {command1}")
        subprocess.run(command1, check=True, shell=True, env=env)
        print(f"Executando: {command2}")
        subprocess.run(command2, check=True, shell=True, env=env)
        print(f"Executando: {command3}")
        subprocess.run(command3, check=True, shell=True, env=env)
        print(f"Executando: {command4}")
        subprocess.run(command4, check=True, shell=True, env=env)
        print("Compilação de todas as versões concluída com sucesso.")
    except subprocess.CalledProcessError as e:
        print(f"Erro durante a compilação: {e}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
