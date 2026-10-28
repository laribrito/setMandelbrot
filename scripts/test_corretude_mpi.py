#!/usr/bin/env python3
"""
==============================================================================
Script de Teste de Corretude para Algoritmos MPI (Mandelbrot)
Disciplina: DEC107 — Processamento Paralelo (UESC)
==============================================================================
Compara a matriz de iterações pixel a pixel de cada versão MPI contra a
saída de referência gerada pelo algoritmo sequencial 'ponto_a_ponto'.
"""

import os
import sys
import glob
import shutil
import subprocess
import time

SCENARIOS = {
    "full": {
        "name": "Visao Completa",
        "RE_MIN": -2.0,
        "RE_MAX": 1.0,
        "IM_MIN": -1.5,
        "IM_MAX": 1.5,
        "MAX_ITER": 1000,
    },
    "zoom": {
        "name": "Zoom (Seahorse Valley)",
        "RE_MIN": -0.745144,
        "RE_MAX": -0.742144,
        "IM_MIN": 0.130326,
        "IM_MAX": 0.133326,
        "MAX_ITER": 1000,
    }
}

def get_root_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir

def write_in_txt(params, root_dir):
    filepath = os.path.join(root_dir, "in.txt")
    with open(filepath, "w") as f:
        for k, v in params.items():
            f.write(f"{k}={v}\n")

def run_cmd(cmd, cwd, timeout=120):
    env = os.environ.copy()
    env.pop("DISPLAY", None)
    res = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout
    )
    return res

def find_latest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def compare_tables(comparador_bin, file_ref, file_test, cwd):
    cmd = [comparador_bin, file_ref, file_test]
    res = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    is_exact = "Sucesso! As duas matrizes sao EXATAMENTE iguais." in res.stdout
    return is_exact, res.stdout

def main():
    root_dir = get_root_dir()
    out_dir = os.path.join(root_dir, "out")
    os.makedirs(out_dir, exist_ok=True)

    comparador_bin = os.path.join(root_dir, "comparador")
    ponto_a_ponto_bin = os.path.join(root_dir, "ponto_a_ponto")

    if not os.path.isfile(comparador_bin) or not os.path.isfile(ponto_a_ponto_bin):
        print("❌ Binários base não encontrados. Compile primeiro com scripts/compile.py.")
        sys.exit(1)

    width = 512
    height = 512

    print("=" * 80)
    print("  SUÍTE DE TESTES DE CORRETUDE - ALGORITMOS MPI (MANDELBROT)")
    print(f"  Resolução de validação: {width}x{height} (262.144 pixels)")
    print("  Comparador: pixel a pixel contra 'ponto_a_ponto' (Golden Reference)")
    print("=" * 80)

    # 1. Gerar Referências (Golden References)
    references = {}
    for sc_key, sc in SCENARIOS.items():
        print(f"\n📌 Gerando matriz de referência para '{sc['name']}'...")
        # Limpa arquivos prévios no out temporariamente
        write_in_txt({
            "WIDTH": width,
            "HEIGHT": height,
            "MAX_ITER": sc["MAX_ITER"],
            "RE_MIN": sc["RE_MIN"],
            "RE_MAX": sc["RE_MAX"],
            "IM_MIN": sc["IM_MIN"],
            "IM_MAX": sc["IM_MAX"],
            "SCHEDULE": "N/A",
            "CHUNK_SIZE": 0,
            "THREADS": 1,
            "SAVE_IMAGE": 1
        }, root_dir)

        t0 = time.time()
        res = run_cmd([ponto_a_ponto_bin], root_dir)
        dt = time.time() - t0
        if res.returncode != 0:
            print(f"❌ Falha ao gerar referência ({sc_key}): {res.stderr}")
            sys.exit(1)

        ref_file = find_latest_file(os.path.join(out_dir, "mandelbrot_pap_*_table.ppm"))
        if not ref_file:
            print(f"❌ Arquivo de tabela não encontrado para referência {sc_key}.")
            sys.exit(1)

        # Salva cópia dedicada da referência
        dedicated_ref = os.path.join(out_dir, f"ref_golden_{sc_key}_table.ppm")
        shutil.copyfile(ref_file, dedicated_ref)
        references[sc_key] = dedicated_ref
        print(f"   ✅ Referência salva: '{os.path.basename(dedicated_ref)}' ({dt:.2f}s)")

    # 2. Configurações de teste para cada versão MPI
    test_configs = [
        # (Nome Legível, Executável, Schedule, Chunks, Lista de Processos, Padrão do Arquivo)
        ("MPI Estático (Block)", "mpi_estatico", "block", [0], [2, 3, 4], "mandelbrot_mpi_estatico_*_table.ppm"),
        ("MPI Estático (Cyclic)", "mpi_estatico", "cyclic", [1, 4, 16], [2, 3, 4], "mandelbrot_mpi_estatico_*_table.ppm"),
        ("MPI Dinâmico Dedicado", "mpi_dinamico", "dynamic", [1, 4, 16], [2, 3, 4], "mandelbrot_mpi_dinamico_chunk*_table.ppm"),
        ("MPI Dinâmico Participativo", "mpi_dinamico_participativo", "dynamic_part", [1, 4, 16], [2, 3, 4], "mandelbrot_mpi_dinamico_part_chunk*_table.ppm"),
        ("MPI Dinâmico Meia-Linha", "mpi_dinamico_meia_linha", "half-row", [1, 2, 8], [2, 3, 4], "mandelbrot_mpi_dinamico_meia_linha_chunk*_table.ppm"),
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    print("\n" + "=" * 80)
    print("  INICIANDO BATERIA DE TESTES DE CORRETUDE")
    print("=" * 80)

    for sc_key in ["full", "zoom"]:
        sc = SCENARIOS[sc_key]
        ref_file = references[sc_key]
        print(f"\n🌐 === CENÁRIO: {sc['name'].upper()} ===")

        for label, exe, sched, chunks, procs_list, file_pattern in test_configs:
            for ch in chunks:
                for np in procs_list:
                    total_tests += 1
                    chunk_desc = f"chunk={ch}" if ch > 0 else "default"
                    test_id = f"[{total_tests:02d}] {label} ({chunk_desc}) - np={np} | {sc_key}"
                    print(f"   {test_id:<55} ... ", end="", flush=True)

                    # Escreve in.txt
                    write_in_txt({
                        "WIDTH": width,
                        "HEIGHT": height,
                        "MAX_ITER": sc["MAX_ITER"],
                        "RE_MIN": sc["RE_MIN"],
                        "RE_MAX": sc["RE_MAX"],
                        "IM_MIN": sc["IM_MIN"],
                        "IM_MAX": sc["IM_MAX"],
                        "SCHEDULE": sched,
                        "CHUNK_SIZE": ch,
                        "THREADS": np,
                        "SAVE_IMAGE": 1
                    }, root_dir)

                    # Executa via mpirun
                    cmd = ["mpirun", "--oversubscribe", "-np", str(np), f"./{exe}"]
                    t0 = time.time()
                    res = run_cmd(cmd, root_dir)
                    dt = time.time() - t0

                    if res.returncode != 0:
                        print(f"❌ CRASH (ret={res.returncode})")
                        failed_tests.append((test_id, f"Processo falhou: {res.stderr[:200]}"))
                        continue

                    # Localiza o arquivo de saída gerado
                    candidate_pattern = os.path.join(out_dir, file_pattern)
                    test_file = find_latest_file(candidate_pattern)

                    if not test_file or os.path.getmtime(test_file) < t0 - 1.0:
                        print("❌ TABELA NÃO ENCONTRADA")
                        failed_tests.append((test_id, "Arquivo de saída não gerado"))
                        continue

                    # Compara com a referência
                    is_exact, comp_out = compare_tables(comparador_bin, ref_file, test_file, root_dir)

                    # Remove arquivos temporários de saída do teste para poupar espaço
                    try:
                        os.remove(test_file)
                        img_cand = test_file.replace("_table.ppm", "_img.ppm")
                        if os.path.isfile(img_cand):
                            os.remove(img_cand)
                    except Exception:
                        pass

                    if is_exact:
                        passed_tests += 1
                        print(f"✅ EXATO (100% idêntico, {dt:.2f}s)")
                    else:
                        print("❌ DIVERGÊNCIA DETECTADA!")
                        diff_summary = "\n".join(comp_out.splitlines()[:6])
                        failed_tests.append((test_id, diff_summary))

    print("\n" + "=" * 80)
    print("  RELATÓRIO CONSOLIDADO DE CORRETUDE MPI")
    print("=" * 80)
    print(f"  Total de Configurações Testadas : {total_tests}")
    print(f"  Total de Sucessos (100% Iguais) : {passed_tests}/{total_tests} ({(passed_tests/total_tests)*100:.1f}%)")
    print(f"  Total de Falhas                 : {len(failed_tests)}")

    if failed_tests:
        print("\n  ❌ FALHAS ENCONTRADAS:")
        for fid, reason in failed_tests:
            print(f"     - {fid}:")
            for line in reason.splitlines():
                print(f"         {line}")
        print("\n⚠️ Alguns algoritmos apresentaram divergências ou falhas.")
        sys.exit(1)
    else:
        print("\n  🏆 SUCESSO TOTAL! Todos os algoritmos MPI produziram matrizes")
        print("     100% idênticas à referência serial ponto_a_ponto em todas as")
        print("     combinações de processos (2, 3, 4) e granularidades de chunk.")
        print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
