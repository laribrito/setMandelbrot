#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import argparse
import csv
import math

# Configurações de regiões do Mandelbrot
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
        "MAX_ITER": 5000,
    }
}

def write_in_txt(params):
    with open("in.txt", "w") as f:
        for k, v in params.items():
            f.write(f"{k}={v}\n")

def compile_binaries():
    print("\n🔨 Compilando executáveis com -O3...")
    env = os.environ.copy()
    if sys.platform == "win32":
        env["PATH"] = "C:\\msys64\\mingw64\\bin;" + env.get("PATH", "")

    cmds = [
        ("ponto_a_ponto", "g++ -O3 -ffast-math -std=c++17 -o ponto_a_ponto ponto_a_ponto.cpp"),
        ("paralelo", "g++ -O3 -ffast-math -fopenmp -std=c++17 -o paralelo paralelo.cpp")
    ]
    for name, cmd in cmds:
        print(f"  -> {cmd}")
        res = subprocess.run(cmd, shell=True, env=env)
        if res.returncode != 0:
            print(f"❌ Erro ao compilar {name}")
            sys.exit(1)
    print("✅ Compilação concluída com sucesso!\n")

def run_executable(executable, machine_name="deCasa"):
    env = os.environ.copy()
    env["HOSTNAME"] = machine_name
    env["COMPUTERNAME"] = machine_name

    cmd = f"./{executable}" if sys.platform != "win32" else f"{executable}.exe"

    start_t = time.time()
    res = subprocess.run(cmd, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    duration = time.time() - start_t

    if res.returncode != 0:
        print(f"    ❌ Erro na execução ({cmd}):\n{res.stderr}")
        return None
    return duration

def load_existing_records(csv_path="dateTimeExecution.csv"):
    """Carrega todos os registros já salvos no CSV de histórico."""
    if not os.path.exists(csv_path):
        return []
    records = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
    except Exception as e:
        print(f"⚠️ Aviso ao ler '{csv_path}': {e}")
    return records

def match_record(row, t, machine_name):
    """Verifica se uma linha do CSV corresponde à configuração de teste especificada."""
    row_mach = row.get("Machine", "").strip().lower()
    if machine_name and row_mach != machine_name.strip().lower():
        return False

    row_code = row.get("Code", "").strip()
    if row_code != t["exe"]:
        return False

    try:
        if int(row.get("WIDTH", 0)) != t["WIDTH"]:
            return False
        if int(row.get("HEIGHT", 0)) != t["HEIGHT"]:
            return False
        if int(row.get("MAX_ITER", 0)) != t["MAX_ITER"]:
            return False
        if not math.isclose(float(row.get("RE_MIN", 0)), t["RE_MIN"], abs_tol=1e-4):
            return False
        if not math.isclose(float(row.get("RE_MAX", 0)), t["RE_MAX"], abs_tol=1e-4):
            return False
        if not math.isclose(float(row.get("IM_MIN", 0)), t["IM_MIN"], abs_tol=1e-4):
            return False
        if not math.isclose(float(row.get("IM_MAX", 0)), t["IM_MAX"], abs_tol=1e-4):
            return False
    except (ValueError, TypeError):
        return False

    if t["exe"] == "paralelo":
        sched = row.get("Schedule", "").strip().lower()
        if sched != t["SCHEDULE"].lower():
            return False
        try:
            chunk = int(row.get("ChunkSize", 0))
            if chunk != t["CHUNK_SIZE"]:
                return False
        except (ValueError, TypeError):
            return False
    else:
        sched = row.get("Schedule", "").strip().upper()
        if sched not in ["N/A", ""]:
            return False

    return True

def get_existing_runs_for_test(t, records, machine_name):
    """Retorna os tempos já registrados no CSV para uma dada configuração."""
    durations = []
    for r in records:
        if match_record(r, t, machine_name):
            try:
                durations.append(float(r["TempoGasto"]))
            except (ValueError, KeyError, TypeError):
                pass
    return durations

def execute_battery(tests, reps, machine_name, force=False):
    existing_records = load_existing_records("dateTimeExecution.csv") if not force else []

    tests_plan = []
    total_runs_needed = 0
    completed_configs = 0

    for t in tests:
        if force:
            already_done_times = []
        else:
            already_done_times = get_existing_runs_for_test(t, existing_records, machine_name)

        count_done = len(already_done_times)
        needed = max(0, reps - count_done)
        total_runs_needed += needed
        if count_done >= reps:
            completed_configs += 1

        tests_plan.append({
            "test": t,
            "existing_times": already_done_times,
            "needed_runs": needed
        })

    print(f"\n==================================================")
    print(f"🚀 INICIANDO BATERIA DE TESTES INTELIGENTE")
    print(f"   Total de Configurações : {len(tests)}")
    print(f"   Configs já concluídas  : {completed_configs}/{len(tests)}")
    print(f"   Repetições por config  : {reps}")
    print(f"   Execuções pendentes    : {total_runs_needed} (de {len(tests) * reps} totais)")
    print(f"   Identificador Máquina  : {machine_name}")
    if force:
        print(f"   ⚠️ Modo FORCE ativado: ignorando histórico do CSV.")
    print(f"==================================================\n")

    if total_runs_needed == 0:
        print("✨ Todas as configurações solicitadas já foram concluídas anteriormente!")
        print("   Nenhuma execução adicional é necessária.")
        print("   (Dica: use --force para reexecutar tudo do zero se desejar).\n")

    results_summary = []
    run_count = 0
    start_total = time.time()

    for idx, item in enumerate(tests_plan, 1):
        t = item["test"]
        existing_times = item["existing_times"]
        needed = item["needed_runs"]
        mode_info = f"{t['SCHEDULE']}:{t['CHUNK_SIZE']}" if t['exe'] == 'paralelo' else "N/A"

        # Se todas as repetições já existem no histórico e não foi solicitado --force
        if needed == 0 and not force:
            avg_prev = sum(existing_times) / len(existing_times) if existing_times else 0.0
            print(f"⏩ [{idx:02d}/{len(tests)}] {t['exe'].upper():13} | {t['scenario_name']:22} | {t['WIDTH']}x{t['HEIGHT']} | Mode: {mode_info:<12} -> [JÁ CONCLUÍDO ({len(existing_times)}/{reps})] (Média: {avg_prev:.2f}s)")
            results_summary.append({
                "exe": t["exe"],
                "scenario": t["scenario_key"],
                "size": f"{t['WIDTH']}x{t['HEIGHT']}",
                "schedule": t["SCHEDULE"],
                "chunk": t["CHUNK_SIZE"],
                "avg_time": avg_prev,
                "reps_executed": len(existing_times),
                "source": "Histórico"
            })
            continue

        status_msg = f"[Executando {needed}/{reps}]" if len(existing_times) == 0 else f"[PARCIAL: {len(existing_times)}/{reps} no histórico, executando +{needed}]"
        print(f"📌 [{idx:02d}/{len(tests)}] {t['exe'].upper():13} | {t['scenario_name']:22} | {t['WIDTH']}x{t['HEIGHT']} | Mode: {mode_info:<12} -> {status_msg}")

        # Grava os parâmetros atuais no in.txt
        write_in_txt({
            "WIDTH": t["WIDTH"],
            "HEIGHT": t["HEIGHT"],
            "MAX_ITER": t["MAX_ITER"],
            "RE_MIN": t["RE_MIN"],
            "RE_MAX": t["RE_MAX"],
            "IM_MIN": t["IM_MIN"],
            "IM_MAX": t["IM_MAX"],
            "SCHEDULE": t["SCHEDULE"],
            "CHUNK_SIZE": t["CHUNK_SIZE"]
        })

        new_durations = []
        for rep in range(1, needed + 1):
            run_count += 1
            cur_rep_num = len(existing_times) + rep
            print(f"   -> Repetição {cur_rep_num}/{reps} (Pendente {run_count}/{total_runs_needed})... ", end="", flush=True)
            d = run_executable(t["exe"], machine_name=machine_name)
            if d is not None:
                new_durations.append(d)
                print(f"OK ({d:.2f}s)")
            else:
                print("FALHOU")

        all_durations = existing_times + new_durations
        avg_t = (sum(all_durations) / len(all_durations)) if all_durations else 0.0
        if all_durations:
            print(f"   ⭐ Média consolidada desta config ({len(all_durations)} execs): {avg_t:.2f}s\n")

        results_summary.append({
            "exe": t["exe"],
            "scenario": t["scenario_key"],
            "size": f"{t['WIDTH']}x{t['HEIGHT']}",
            "schedule": t["SCHEDULE"],
            "chunk": t["CHUNK_SIZE"],
            "avg_time": avg_t,
            "reps_executed": len(all_durations),
            "source": "Novo" if not existing_times else "Misto"
        })

    total_time = time.time() - start_total
    print("==================================================")
    print(f"🎉 Bateria de testes concluída em {total_time/60:.2f} minutos ({total_time:.1f}s)!")
    print(f"📊 Todos os registros detalhados estão em 'dateTimeExecution.csv'.\n")

    # Imprime tabela resumo consolidada
    print("📋 RESUMO CONSOLIDADO (MÉDIAS):")
    print(f"{'Programa':<14} | {'Cenário':<6} | {'Resolução':<11} | {'Schedule':<9} | {'Chunk':<6} | {'Execuções':<10} | {'Tempo Médio':<11}")
    print("-" * 88)
    for r in results_summary:
        print(f"{r['exe']:<14} | {r['scenario']:<6} | {r['size']:<11} | {r['schedule']:<9} | {str(r['chunk']):<6} | {str(r['reps_executed']) + '/' + str(reps):<10} | {r['avg_time']:.2f}s")
    print("=" * 88)

def build_test_list(mode, scenarios, sizes, schedules):
    tests = []
    # 1. Ponto a Ponto
    if mode in ["all", "pap"]:
        for scen_key in scenarios:
            scen = SCENARIOS[scen_key]
            for size in sizes:
                tests.append({
                    "exe": "ponto_a_ponto",
                    "scenario_key": scen_key,
                    "scenario_name": scen["name"],
                    "WIDTH": size,
                    "HEIGHT": size,
                    "MAX_ITER": scen["MAX_ITER"],
                    "RE_MIN": scen["RE_MIN"],
                    "RE_MAX": scen["RE_MAX"],
                    "IM_MIN": scen["IM_MIN"],
                    "IM_MAX": scen["IM_MAX"],
                    "SCHEDULE": "N/A",
                    "CHUNK_SIZE": 0
                })

    # 2. Paralelo
    if mode in ["all", "paralelo"]:
        for scen_key in scenarios:
            scen = SCENARIOS[scen_key]
            for size in sizes:
                for sched_spec in schedules:
                    sched_name, chunk_str = sched_spec.split(":")
                    tests.append({
                        "exe": "paralelo",
                        "scenario_key": scen_key,
                        "scenario_name": scen["name"],
                        "WIDTH": size,
                        "HEIGHT": size,
                        "MAX_ITER": scen["MAX_ITER"],
                        "RE_MIN": scen["RE_MIN"],
                        "RE_MAX": scen["RE_MAX"],
                        "IM_MIN": scen["IM_MIN"],
                        "IM_MAX": scen["IM_MAX"],
                        "SCHEDULE": sched_name,
                        "CHUNK_SIZE": int(chunk_str)
                    })
    return tests

def show_interactive_menu():
    print("\n=======================================================")
    print("        ⚙️  AUTOMATIZADOR DE TESTES MANDELBROT         ")
    print("=======================================================")
    print(" Escolha uma opção para executar:")
    print("  [1] 🚀 RODAR TUDO UM APÓS O OUTRO (Ponto a Ponto + 4 Schedules Paralelos)")
    print("      -> Resoluções: 4096, 8192, 16384 | Full e Zoom | 3x cada")
    print("  [2] 🏎️  Rodar apenas PONTO A PONTO (3x cada em 4096, 8192, 16384)")
    print("  [3] ⚡ Rodar apenas PARALELO (Static, Dynamic, Guided, Auto - 3x cada)")
    print("  [4] 🧪 Teste Rápido (Apenas resolução 4096, 1 repetição de cada)")
    print("  [0] Sair")
    print("=======================================================")
    choice = input("Digite a opção desejada [1-4, 0]: ").strip()
    return choice

def main():
    parser = argparse.ArgumentParser(description="Automatizador de Benchmarks Mandelbrot")
    parser.add_argument("--tudo", "-t", action="store_true",
                        help="Opção rápida: roda TUDO um após o outro (ponto a ponto + paralelo, 3x cada)")
    parser.add_argument("--mode", choices=["all", "pap", "paralelo"], default=None,
                        help="Quais programas rodar: 'pap', 'paralelo' ou 'all'")
    parser.add_argument("--reps", type=int, default=3,
                        help="Número de repetições por teste. Padrão: 3")
    parser.add_argument("--sizes", nargs="+", type=int, default=[4096, 8192, 16384],
                        help="Resoluções a testar. Padrão: 4096 8192 16384")
    parser.add_argument("--scenarios", nargs="+", choices=["full", "zoom"], default=["full", "zoom"],
                        help="Cenários a testar: 'full', 'zoom' ou ambos. Padrão: full zoom")
    parser.add_argument("--schedules", nargs="+", default=["static:64", "dynamic:1", "dynamic:64", "guided:64", "auto:0"],
                        help="Schedules para o paralelo modo:chunk (ex: dynamic:1 static:64 guided:64 auto:0)")
    parser.add_argument("--no-compile", action="store_true", help="Pular etapa de compilação")
    parser.add_argument("--machine", default="deCasa", help="Nome da máquina para o CSV (padrão: deCasa)")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Forçar a execução de todos os testes ignorando o histórico prévio")

    args = parser.parse_args()

    # Se chamado com --tudo, roda toda a suíte automaticamente
    if args.tudo:
        if not args.no_compile:
            compile_binaries()
        tests = build_test_list("all", args.scenarios, args.sizes, args.schedules)
        execute_battery(tests, args.reps, args.machine, force=args.force)
        return

    # Se nenhum argumento de modo for passado pela CLI, abre o menu interativo
    if args.mode is None:
        choice = show_interactive_menu()
        if choice == "1":
            mode = "all"
            sizes = [4096, 8192, 16384]
            reps = 3
        elif choice == "2":
            mode = "pap"
            sizes = [4096, 8192, 16384]
            reps = 3
        elif choice == "3":
            mode = "paralelo"
            sizes = [4096, 8192, 16384]
            reps = 3
        elif choice == "4":
            mode = "all"
            sizes = [4096]
            reps = 1
        elif choice == "0":
            print("Saindo...")
            sys.exit(0)
        else:
            print("Opção inválida!")
            sys.exit(1)

        if not args.no_compile:
            compile_binaries()
        tests = build_test_list(mode, args.scenarios, sizes, args.schedules)
        execute_battery(tests, reps, args.machine, force=args.force)
        return

    # Execução via argumentos de linha de comando
    if not args.no_compile:
        compile_binaries()
    tests = build_test_list(args.mode, args.scenarios, args.sizes, args.schedules)
    execute_battery(tests, args.reps, args.machine, force=args.force)

if __name__ == "__main__":
    main()
