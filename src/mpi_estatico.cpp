#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <vector>
#include <cstdlib>
#include <filesystem>
#include <algorithm>
#include <mpi.h>

/**
 *
 * Estratégias suportadas via 'in.txt' (SCHEDULE):
 *  1. 'block' ou 'static': Cada processo recebe um bloco contínuo de linhas (H/P).
 *     Comunicação final via MPI_Gatherv. Simples, mas suscetível a desbalanceamento
 *     de carga quando nós processam a região central (cardióide).
 *
 *  2. 'cyclic' ou 'intercalado': Distribuição round-robin de linhas (j % P == rank).
 *     Ameniza drasticamente o desbalanceamento estático de carga, pois todas as
 *     regiões (rápidas e lentas) são intercaladas entre os processos.
 */

class Complex {
public:
    double real;
    double imag;
    const double EPSILON = 1e-9;

    Complex(double r = 0.0, double i = 0.0) : real(r), imag(i) {}

    Complex operator+(const Complex& other) const {
        return Complex(real + other.real, imag + other.imag);
    }

    Complex operator*(const Complex& other) const {
        return Complex(real * other.real - imag * other.imag,
                       real * other.imag + imag * other.real);
    }

    Complex& operator=(const Complex& other) {
        real = other.real;
        imag = other.imag;
        return *this;
    }

    Complex& operator=(double val) {
        real = val;
        imag = 0.0;
        return *this;
    }

    double magnitudeSquared() const {
        return real * real + imag * imag;
    }
};

struct RGB {
    int r, g, b;
};

// Retorna a cor da paleta clássica baseada no valor t_base entre 0.0 e 1.0
RGB getPaletteColor(double t_base) {
    double t = std::cbrt(t_base);
    int r = static_cast<int>(9   * (1-t) * t*t*t        * 255);
    int g = static_cast<int>(15  * (1-t)*(1-t) * t*t    * 255);
    int b = static_cast<int>(8.5 * (1-t)*(1-t)*(1-t)*t  * 255);
    return {r, g, b};
}

int main(int argc, char* argv[]) {
    // Inicialização do ambiente MPI
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parâmetros padrão
    int WIDTH = 4096, HEIGHT = 4096, MAX_ITER = 1000;
    double RE_MIN = -2.0, RE_MAX = 1.0, IM_MIN = -1.5, IM_MAX = 1.5;
    std::string schedName = "block"; // 'block'/'static' ou 'cyclic'/'intercalado'
    int chunkSize = 0;
    bool saveImage = false; // Desativado para testes de desempenho

    // Apenas o Rank 0 lê o arquivo in.txt para evitar contenção de I/O
    if (rank == 0) {
        std::ifstream inFile("in.txt");
        if (inFile.is_open()) {
            std::string line;
            while (std::getline(inFile, line)) {
                if (line.empty() || line[0] == '#') continue;
                size_t pos = line.find('=');
                if (pos != std::string::npos) {
                    std::string key   = line.substr(0, pos);
                    std::string value = line.substr(pos + 1);

                    if      (key == "WIDTH")      WIDTH      = std::stoi(value);
                    else if (key == "HEIGHT")     HEIGHT     = std::stoi(value);
                    else if (key == "MAX_ITER")   MAX_ITER   = std::stoi(value);
                    else if (key == "RE_MIN")     RE_MIN     = std::stod(value);
                    else if (key == "RE_MAX")     RE_MAX     = std::stod(value);
                    else if (key == "IM_MIN")     IM_MIN     = std::stod(value);
                    else if (key == "IM_MAX")     IM_MAX     = std::stod(value);
                    else if (key == "SCHEDULE")   schedName  = value;
                    else if (key == "CHUNK_SIZE") chunkSize  = std::stoi(value);
                    else if (key == "SAVE_IMAGE") saveImage  = (std::stoi(value) != 0);
                }
            }
            inFile.close();
        } else {
            std::cerr << "[Rank 0] Aviso: Nao foi possivel abrir in.txt. Utilizando parametros padroes.\n";
        }
    }

    // Normaliza nome do escalonamento estático
    // Transmite os parâmetros para todos os processos via Broadcast
    int intParams[4] = {WIDTH, HEIGHT, MAX_ITER, chunkSize};
    MPI_Bcast(intParams, 4, MPI_INT, 0, MPI_COMM_WORLD);
    WIDTH     = intParams[0];
    HEIGHT    = intParams[1];
    MAX_ITER  = intParams[2];
    chunkSize = intParams[3];

    double doubleParams[4] = {RE_MIN, RE_MAX, IM_MIN, IM_MAX};
    MPI_Bcast(doubleParams, 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    RE_MIN = doubleParams[0];
    RE_MAX = doubleParams[1];
    IM_MIN = doubleParams[2];
    IM_MAX = doubleParams[3];

    char schedBuffer[64];
    if (rank == 0) {
        std::snprintf(schedBuffer, sizeof(schedBuffer), "%s", schedName.c_str());
    }
    MPI_Bcast(schedBuffer, 64, MPI_CHAR, 0, MPI_COMM_WORLD);
    schedName = schedBuffer;

    // Converte para minúsculas
    std::transform(schedName.begin(), schedName.end(), schedName.begin(), ::tolower);
    bool isCyclic = (schedName == "cyclic" || schedName == "intercalado");
    if (!isCyclic && schedName != "block" && schedName != "static") {
        if (rank == 0) {
            std::cerr << "[Rank 0] Aviso: SCHEDULE='" << schedName << "' nao e 'block' ou 'cyclic'. Usando 'block'.\n";
        }
        schedName = "block";
    }

    const double tamPixel_re = (RE_MAX - RE_MIN) / WIDTH;
    const double tamPixel_im = (IM_MAX - IM_MIN) / HEIGHT;

    // Matriz global completa (alocada no Rank 0 para receber os dados coletados)
    std::vector<int> temps;
    if (rank == 0) {
        temps.resize(HEIGHT * WIDTH, 0);
        std::cout << "========================================================\n"
                  << "  Benchmark Mandelbrot - MPI Estatico\n"
                  << "  Processos MPI: " << size << "\n"
                  << "  Estrategia   : " << (isCyclic ? "Ciclica (Intercalada)" : "Blocos Contiguos") << "\n"
                  << "  Resolucao    : " << WIDTH << "x" << HEIGHT << " (" << MAX_ITER << " max iter)\n"
                  << "========================================================\n" << std::flush;
    }

    // Configuração da partição de trabalho
    std::vector<int> rows_per_rank(size, 0);
    std::vector<int> start_row_per_rank(size, 0);
    std::vector<int> recvcounts(size, 0);
    std::vector<int> displs(size, 0);

    std::vector<int> local_buffer;
    int my_rows = 0;

    if (!isCyclic) {
        // --- ESTRATÉGIA 1: BLOCOS CONTÍGUOS ---
        int base_rows = HEIGHT / size;
        int rem = HEIGHT % size;
        int current_start = 0;
        for (int r = 0; r < size; ++r) {
            rows_per_rank[r] = base_rows + (r < rem ? 1 : 0);
            start_row_per_rank[r] = current_start;
            recvcounts[r] = rows_per_rank[r] * WIDTH;
            displs[r] = current_start * WIDTH;
            current_start += rows_per_rank[r];
        }
        my_rows = rows_per_rank[rank];
        local_buffer.resize(my_rows * WIDTH);
    } else {
        // --- ESTRATÉGIA 2: CÍCLICA / INTERCALADA COM SUPORTE A CHUNK_SIZE (BLOCK-CYCLIC) ---
        int stepChunk = (chunkSize > 0) ? chunkSize : 1;
        int total_chunks = (HEIGHT + stepChunk - 1) / stepChunk;

        int current_displ = 0;
        for (int r = 0; r < size; ++r) {
            int count = 0;
            for (int c = r; c < total_chunks; c += size) {
                int start_line = c * stepChunk;
                count += std::min(stepChunk, HEIGHT - start_line);
            }
            rows_per_rank[r] = count;
            recvcounts[r] = count * WIDTH;
            displs[r] = current_displ;
            current_displ += recvcounts[r];
        }
        my_rows = rows_per_rank[rank];
        local_buffer.resize(my_rows * WIDTH);
    }

    // Barreira inicial para sincronização rigorosa antes da cronometragem
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // Tempo de cálculo local (apenas computação dos pixels, antes de esperar no Gather)
    double my_compute_time = 0.0;

    if (!isCyclic) {
        // Blocos Contíguos
        int my_start = start_row_per_rank[rank];
        for (int lr = 0; lr < my_rows; ++lr) {
            int j = my_start + lr;
            double y = IM_MAX - j * tamPixel_im;
            for (int i = 0; i < WIDTH; ++i) {
                double x = RE_MIN + i * tamPixel_re;
                Complex c(x, y);
                Complex z = 0;
                int temp = 0;

                while (z.magnitudeSquared() < 4.0 && temp < MAX_ITER) {
                    z = z * z + c;
                    temp++;
                }
                local_buffer[lr * WIDTH + i] = temp;
            }
        }
        my_compute_time = MPI_Wtime() - t_start;

        // Coleta direta no Rank 0 via MPI_Gatherv
        MPI_Gatherv(local_buffer.data(), my_rows * WIDTH, MPI_INT,
                    temps.data(), recvcounts.data(), displs.data(), MPI_INT,
                    0, MPI_COMM_WORLD);

    } else {
        // Distribuição Cíclica / Block-Cyclic (chunks de stepChunk linhas)
        int stepChunk = (chunkSize > 0) ? chunkSize : 1;
        int total_chunks = (HEIGHT + stepChunk - 1) / stepChunk;

        int lr = 0;
        for (int c = rank; c < total_chunks; c += size) {
            int start_line = c * stepChunk;
            int num_lines  = std::min(stepChunk, HEIGHT - start_line);

            for (int line_idx = 0; line_idx < num_lines; ++line_idx) {
                int j = start_line + line_idx;
                double y = IM_MAX - j * tamPixel_im;
                for (int i = 0; i < WIDTH; ++i) {
                    double x = RE_MIN + i * tamPixel_re;
                    Complex comp(x, y);
                    Complex z = 0;
                    int temp = 0;

                    while (z.magnitudeSquared() < 4.0 && temp < MAX_ITER) {
                        z = z * z + comp;
                        temp++;
                    }
                    local_buffer[lr * WIDTH + i] = temp;
                }
                lr++;
            }
        }
        my_compute_time = MPI_Wtime() - t_start;

        // Buffer temporário para receber os blocos cíclicos contíguos no Rank 0
        std::vector<int> cyclic_recv;
        if (rank == 0) {
            cyclic_recv.resize(HEIGHT * WIDTH);
        }

        MPI_Gatherv(local_buffer.data(), my_rows * WIDTH, MPI_INT,
                    cyclic_recv.data(), recvcounts.data(), displs.data(), MPI_INT,
                    0, MPI_COMM_WORLD);

        // Rank 0 desentrelaça as linhas para reconstituir a imagem na ordem 0..HEIGHT-1
        if (rank == 0) {
            for (int r = 0; r < size; ++r) {
                int r_offset = displs[r];
                int lr_r = 0;
                for (int c = r; c < total_chunks; c += size) {
                    int start_line = c * stepChunk;
                    int num_lines  = std::min(stepChunk, HEIGHT - start_line);

                    for (int line_idx = 0; line_idx < num_lines; ++line_idx) {
                        int original_j = start_line + line_idx;
                        std::copy(cyclic_recv.begin() + r_offset + lr_r * WIDTH,
                                  cyclic_recv.begin() + r_offset + (lr_r + 1) * WIDTH,
                                  temps.begin() + original_j * WIDTH);
                        lr_r++;
                    }
                }
            }
        }
    }

    // Fim da medição de tempo total da etapa paralela
    double t_end = MPI_Wtime();
    double local_elapsed = t_end - t_start;

    // Redução para obter o tempo máximo entre todos os processos (métrica formal paralela)
    double max_elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Coleta dos tempos individuais de cálculo de todos os processos no Rank 0
    std::vector<double> all_proc_times(size, 0.0);
    MPI_Gather(&my_compute_time, 1, MPI_DOUBLE, all_proc_times.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // -------------------------------------------------------------------------
    // SALVAMENTO DE RESULTADOS E LOGS (APENAS RANK 0)
    // -------------------------------------------------------------------------
    if (rank == 0) {
        std::cout << "Tempo gasto (processamento + comunicacao): " << max_elapsed << " segundos.\n";

        // Identifica caminho do CSV oficial do benchmark
        std::string csvPath = "data/dateTimeExecution.csv";
        if (!std::filesystem::exists("data/dateTimeExecution.csv") && std::filesystem::exists("dateTimeExecution.csv")) {
            csvPath = "dateTimeExecution.csv";
        } else {
            std::filesystem::create_directories("data");
        }

        // Geração de imagem e tabela (cancelada durante os testes para evitar gargalo de I/O em disco)
        if (saveImage) {
            std::filesystem::create_directories("out");
            int executionNum = 1;
            std::ifstream checkFile(csvPath);
            if (checkFile.is_open()) {
                std::string line;
                while (std::getline(checkFile, line)) {
                    if (!line.empty()) executionNum++;
                }
                checkFile.close();
            }

            std::string prefix = "out/mandelbrot_mpi_estatico_" + schedName + "_" + std::to_string(executionNum);

            // 1. Grava Tabela de Iterações (compatível com comparador.cpp)
            std::string tableFilename = prefix + "_table.ppm";
            std::ofstream tableFile(tableFilename);
            if (tableFile.is_open()) {
                for (int j = 0; j < HEIGHT; ++j) {
                    for (int i = 0; i < WIDTH; ++i) {
                        tableFile << temps[j * WIDTH + i] << (i == WIDTH - 1 ? "" : " ");
                    }
                    tableFile << "\n";
                }
                tableFile.close();
                std::cout << "Tabela de iteracoes salva em '" << tableFilename << "'\n";
            }

            // 2. Grava Imagem PPM com gradiente de legenda
            std::string ppmFilename = prefix + "_img.ppm";
            std::ofstream ppmFile(ppmFilename);
            if (ppmFile.is_open()) {
                int blackLineWidth = static_cast<int>(WIDTH * 0.01);
                int gradientWidth  = static_cast<int>(WIDTH * 0.10);
                int ppmWidth = WIDTH + blackLineWidth + gradientWidth;

                ppmFile << "P3\n" << ppmWidth << " " << HEIGHT << "\n255\n";
                for (int j = 0; j < HEIGHT; ++j) {
                    for (int i = 0; i < WIDTH; ++i) {
                        int val = temps[j * WIDTH + i];
                        if (val >= MAX_ITER) {
                            ppmFile << "0 0 0 ";
                        } else {
                            RGB color = getPaletteColor(static_cast<double>(val) / MAX_ITER);
                            ppmFile << color.r << " " << color.g << " " << color.b << " ";
                        }
                    }
                    for (int i = 0; i < blackLineWidth; ++i) ppmFile << "0 0 0 ";
                    double t_grad = static_cast<double>(HEIGHT - 1 - j) / (HEIGHT - 1);
                    RGB gradColor = getPaletteColor(t_grad);
                    for (int i = 0; i < gradientWidth; ++i) {
                        ppmFile << gradColor.r << " " << gradColor.g << " " << gradColor.b << " ";
                    }
                    ppmFile << "\n";
                }
                ppmFile.close();
                std::cout << "Imagem salva com sucesso em '" << ppmFilename << "'\n";
            }
        }

        // 3. Registra execução no CSV oficial do benchmark
        bool isNewFile = false;
        {
            std::ifstream testFile(csvPath);
            if (!testFile || testFile.peek() == std::ifstream::traits_type::eof()) {
                isNewFile = true;
            }
        }

        std::ofstream csvFile(csvPath, std::ios::app);
        if (csvFile.is_open()) {
            const char* machineName = std::getenv("COMPUTERNAME");
            if (!machineName) machineName = std::getenv("HOSTNAME");
            if (!machineName) machineName = "Unknown";

            if (isNewFile) {
                csvFile << "DataHora,TempoGasto,WIDTH,HEIGHT,MAX_ITER,RE_MIN,RE_MAX,IM_MIN,IM_MAX,Machine,Code,Schedule,ChunkSize,Threads\n";
            }

            auto now = std::chrono::system_clock::now();
            std::time_t current_time = std::chrono::system_clock::to_time_t(now);

            csvFile << std::put_time(std::localtime(&current_time), "%Y-%m-%d %H:%M:%S")
                    << "," << max_elapsed
                    << "," << WIDTH
                    << "," << HEIGHT
                    << "," << MAX_ITER
                    << "," << RE_MIN
                    << "," << RE_MAX
                    << "," << IM_MIN
                    << "," << IM_MAX
                    << "," << machineName
                    << ",mpi_estatico"
                    << "," << schedName
                    << "," << chunkSize
                    << "," << size
                    << "\n";
            csvFile.close();
            std::cout << "Registro salvo em '" << csvPath << "'\n";
        }

        // 4. Registra tempos individuais de cada processo para cálculo de desbalanceamento
        std::string procCsvPath = "data/tempos_processos_mpi.csv";
        bool isNewProcFile = false;
        {
            std::ifstream testProc(procCsvPath);
            if (!testProc || testProc.peek() == std::ifstream::traits_type::eof()) {
                isNewProcFile = true;
            }
        }
        std::ofstream procCsvFile(procCsvPath, std::ios::app);
        if (procCsvFile.is_open()) {
            if (isNewProcFile) {
                procCsvFile << "DataHora,Code,Schedule,ChunkSize,Threads,WIDTH,HEIGHT,MAX_ITER,Rank,TempoProcesso\n";
            }
            auto now = std::chrono::system_clock::now();
            std::time_t current_time = std::chrono::system_clock::to_time_t(now);
            for (int r = 0; r < size; ++r) {
                procCsvFile << std::put_time(std::localtime(&current_time), "%Y-%m-%d %H:%M:%S")
                            << ",mpi_estatico," << schedName << "," << chunkSize << "," << size
                            << "," << WIDTH << "," << HEIGHT << "," << MAX_ITER
                            << "," << r << "," << all_proc_times[r] << "\n";
            }
            procCsvFile.close();
        }

        std::cout << "Execucao MPI Estatica concluida com exito!\n";
    }

    MPI_Finalize();
    return 0;
}
