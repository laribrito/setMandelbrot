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

RGB getPaletteColor(double t_base) {
    double t = std::cbrt(t_base);
    int r = static_cast<int>(9   * (1-t) * t*t*t        * 255);
    int g = static_cast<int>(15  * (1-t)*(1-t) * t*t    * 255);
    int b = static_cast<int>(8.5 * (1-t)*(1-t)*(1-t)*t  * 255);
    return {r, g, b};
}

enum MsgTags {
    TAG_WORK   = 1,
    TAG_RESULT = 2,
    TAG_STOP   = 3
};

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int WIDTH = 4096, HEIGHT = 4096, MAX_ITER = 1000;
    double RE_MIN = -2.0, RE_MAX = 1.0, IM_MIN = -1.5, IM_MAX = 1.5;
    int chunkSize = 1; // Quantidade de meias-linhas por tarefa
    bool saveImage = false; // Desativado para testes de desempenho

    // Apenas o Rank 0 lê in.txt
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
                    else if (key == "CHUNK_SIZE") chunkSize  = std::stoi(value);
                    else if (key == "SAVE_IMAGE") saveImage  = (std::stoi(value) != 0);
                }
            }
            inFile.close();
        } else {
            std::cerr << "[Rank 0] Aviso: in.txt nao encontrado. Usando padroes.\n";
        }

        if (chunkSize <= 0) {
            chunkSize = 1;
        }
    }

    // Transmite parâmetros via Broadcast
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

    const double tamPixel_re = (RE_MAX - RE_MIN) / WIDTH;
    const double tamPixel_im = (IM_MAX - IM_MIN) / HEIGHT;

    // Total de meias-linhas na imagem
    const int total_halves = 2 * HEIGHT;

    // Capacidade máxima de buffer (chunkSize meias-linhas)
    const int max_half_width = WIDTH - (WIDTH / 2);
    const int max_buffer_capacity = 2 + chunkSize * max_half_width;

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // -------------------------------------------------------------------------
    // CASO ESPECIAL: 1 PROCESSO (SERIAL)
    // -------------------------------------------------------------------------
    if (size == 1) {
        std::vector<int> temps(HEIGHT * WIDTH, 0);
        for (int j = 0; j < HEIGHT; ++j) {
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
                temps[j * WIDTH + i] = temp;
            }
        }
        double elapsed = MPI_Wtime() - t_start;
        std::cout << "[MPI Meia-Linha (1 proc)] Tempo gasto: " << elapsed << " s\n";
        MPI_Finalize();
        return 0;
    }

    // -------------------------------------------------------------------------
    // COORDENADOR (RANK 0): Despacha tarefas de chunkSize meias-linhas
    // -------------------------------------------------------------------------
    if (rank == 0) {
        std::cout << "========================================================\n"
                  << "  Benchmark Mandelbrot - MPI Dinamico (MEIA-LINHA)\n"
                  << "  Processos Totais : " << size << " (1 Coordenador + " << (size - 1) << " Trabalhadores)\n"
                  << "  Granularidade    : " << chunkSize << " meia-linha(s) por tarefa\n"
                  << "  Total Meias-Linhas: " << total_halves << "\n"
                  << "  Resolucao        : " << WIDTH << "x" << HEIGHT << " (" << MAX_ITER << " max iter)\n"
                  << "========================================================\n" << std::flush;

        std::vector<int> temps(HEIGHT * WIDTH, 0);

        int next_half_index = 0;
        int active_workers = 0;

        // 1. Envia o primeiro lote de meias-linhas para cada trabalhador
        for (int worker = 1; worker < size; ++worker) {
            if (next_half_index < total_halves) {
                int halves_to_send = std::min(chunkSize, total_halves - next_half_index);
                int task[2] = {next_half_index, halves_to_send};
                MPI_Send(task, 2, MPI_INT, worker, TAG_WORK, MPI_COMM_WORLD);
                next_half_index += halves_to_send;
                active_workers++;
            } else {
                int stop_signal[2] = {-1, 0};
                MPI_Send(stop_signal, 2, MPI_INT, worker, TAG_STOP, MPI_COMM_WORLD);
            }
        }

        std::vector<int> recv_buffer(max_buffer_capacity);

        // 2. Loop de atendimento dinâmico
        while (active_workers > 0) {
            MPI_Status status;
            MPI_Recv(recv_buffer.data(), recv_buffer.size(), MPI_INT,
                     MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &status);

            int worker     = status.MPI_SOURCE;
            int start_half = recv_buffer[0];
            int num_halves = recv_buffer[1];

            // Desempacota cada meia-linha calculada no lote
            int offset = 2;
            for (int h = 0; h < num_halves; ++h) {
                int half_id    = start_half + h;
                int row        = half_id / 2;
                int which_half = half_id % 2;
                int col_start  = (which_half == 0) ? 0 : (WIDTH / 2);
                int num_cols   = (which_half == 0) ? (WIDTH / 2) : (WIDTH - WIDTH / 2);

                std::copy(recv_buffer.begin() + offset,
                          recv_buffer.begin() + offset + num_cols,
                          temps.begin() + row * WIDTH + col_start);
                offset += num_cols;
            }

            // Despacha próximo lote de meias-linhas ou dispensa o trabalhador
            if (next_half_index < total_halves) {
                int halves_to_send = std::min(chunkSize, total_halves - next_half_index);
                int task[2] = {next_half_index, halves_to_send};
                MPI_Send(task, 2, MPI_INT, worker, TAG_WORK, MPI_COMM_WORLD);
                next_half_index += halves_to_send;
            } else {
                int stop_signal[2] = {-1, 0};
                MPI_Send(stop_signal, 2, MPI_INT, worker, TAG_STOP, MPI_COMM_WORLD);
                active_workers--;
            }
        }

        double t_end = MPI_Wtime();
        double elapsed = t_end - t_start;

        std::cout << "Tempo gasto (Meia-Linha total): " << elapsed << " segundos.\n";

        // Grava saídas
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

            std::string prefix = "out/mandelbrot_mpi_dinamico_meia_linha_chunk" + std::to_string(chunkSize) + "_" + std::to_string(executionNum);

            // 1. Grava Tabela de Iterações
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
                std::cout << "Tabela salva em '" << tableFilename << "'\n";
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
                std::cout << "Imagem salva em '" << ppmFilename << "'\n";
            }
        }

        // 3. Registra no CSV oficial do benchmark
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
                    << "," << elapsed
                    << "," << WIDTH
                    << "," << HEIGHT
                    << "," << MAX_ITER
                    << "," << RE_MIN
                    << "," << RE_MAX
                    << "," << IM_MIN
                    << "," << IM_MAX
                    << "," << machineName
                    << ",mpi_dinamico_meia_linha"
                    << ",half-row"
                    << "," << chunkSize
                    << "," << size
                    << "\n";
            csvFile.close();
            std::cout << "Registro salvo em '" << csvPath << "'\n";
        }

        // Coordenador dedicado não computa pixels (tempo = 0.0)
        double my_compute_time = 0.0;
        std::vector<double> all_proc_times(size, 0.0);
        MPI_Gather(&my_compute_time, 1, MPI_DOUBLE, all_proc_times.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

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
                            << ",mpi_dinamico_meia_linha,half-row," << chunkSize << "," << size
                            << "," << WIDTH << "," << HEIGHT << "," << MAX_ITER
                            << "," << r << "," << all_proc_times[r] << "\n";
            }
            procCsvFile.close();
        }

        std::cout << "Execucao MPI Meia-Linha finalizada com exito!\n";

    } else {
        // ---------------------------------------------------------------------
        // TRABALHADORES (RANKS 1 A SIZE-1)
        // ---------------------------------------------------------------------
        std::vector<int> send_buffer(max_buffer_capacity);
        double my_compute_time = 0.0;

        while (true) {
            int task[2]; // [start_half, num_halves]
            MPI_Status status;
            MPI_Recv(task, 2, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_STOP || task[0] < 0) {
                break;
            }

            int start_half = task[0];
            int num_halves = task[1];

            send_buffer[0] = start_half;
            send_buffer[1] = num_halves;

            double t_w0 = MPI_Wtime();
            int offset = 2;
            for (int h = 0; h < num_halves; ++h) {
                int half_id    = start_half + h;
                int row        = half_id / 2;
                int which_half = half_id % 2;
                int col_start  = (which_half == 0) ? 0 : (WIDTH / 2);
                int num_cols   = (which_half == 0) ? (WIDTH / 2) : (WIDTH - WIDTH / 2);

                double y = IM_MAX - row * tamPixel_im;

                for (int c = 0; c < num_cols; ++c) {
                    int i = col_start + c;
                    double x = RE_MIN + i * tamPixel_re;
                    Complex comp(x, y);
                    Complex z = 0;
                    int temp = 0;

                    while (z.magnitudeSquared() < 4.0 && temp < MAX_ITER) {
                        z = z * z + comp;
                        temp++;
                    }
                    send_buffer[offset + c] = temp;
                }
                offset += num_cols;
            }
            my_compute_time += (MPI_Wtime() - t_w0);

            // Envia resultado do lote de meias-linhas
            MPI_Send(send_buffer.data(), offset, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
        }

        // Envia seu tempo total de computação para o Rank 0
        MPI_Gather(&my_compute_time, 1, MPI_DOUBLE, nullptr, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
