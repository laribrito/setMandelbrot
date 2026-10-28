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

// Retorna a cor da paleta clássica baseada no valor t_base entre 0.0 e 1.0
RGB getPaletteColor(double t_base) {
    double t = std::cbrt(t_base);
    int r = static_cast<int>(9   * (1-t) * t*t*t        * 255);
    int g = static_cast<int>(15  * (1-t)*(1-t) * t*t    * 255);
    int b = static_cast<int>(8.5 * (1-t)*(1-t)*(1-t)*t  * 255);
    return {r, g, b};
}

// Tags de comunicação MPI
enum MsgTags {
    TAG_WORK   = 1, // Envio de fatia de trabalho do coordenador para o trabalhador
    TAG_RESULT = 2, // Retorno dos pixels calculados do trabalhador para o coordenador
    TAG_STOP   = 3  // Sinalização de fim de tarefas
};

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parâmetros padrão
    int WIDTH = 4096, HEIGHT = 4096, MAX_ITER = 1000;
    double RE_MIN = -2.0, RE_MAX = 1.0, IM_MIN = -1.5, IM_MAX = 1.5;
    std::string schedName = "dynamic_participativo";
    int chunkSize = 1; // Granularidade em linhas (padrão 1 ou configurado em in.txt)
    bool saveImage = false; // Desativado para testes de desempenho

    // Apenas o Rank 0 lê o arquivo in.txt
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
            std::cerr << "[Rank 0] Aviso: Nao foi possivel abrir in.txt. Utilizando parametros padroes.\n";
        }

        if (chunkSize <= 0) {
            chunkSize = 1;
        }
    }

    // Transmite configurações a todos os nós
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

    // Sincronização geral antes de iniciar a medição do tempo
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // -------------------------------------------------------------------------
    // CASO ESPECIAL: EXECUÇÃO COM 1 PROCESSO
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

        double t_end = MPI_Wtime();
        double elapsed = t_end - t_start;

        std::cout << "[MPI Coordenador-Participativo (1 proc)] Tempo gasto: " << elapsed << " segundos.\n";
        std::filesystem::create_directories("out");
        std::filesystem::create_directories("data");
        std::ofstream csvFile("data/dateTimeExecution.csv", std::ios::app);
        if (csvFile.is_open()) {
            const char* machineName = std::getenv("COMPUTERNAME");
            if (!machineName) machineName = std::getenv("HOSTNAME");
            if (!machineName) machineName = "Unknown";
            auto now = std::chrono::system_clock::now();
            std::time_t current_time = std::chrono::system_clock::to_time_t(now);
            csvFile << std::put_time(std::localtime(&current_time), "%Y-%m-%d %H:%M:%S")
                    << "," << elapsed
                    << "," << WIDTH << "," << HEIGHT << "," << MAX_ITER
                    << "," << RE_MIN << "," << RE_MAX << "," << IM_MIN << "," << IM_MAX
                    << "," << machineName << ",mpi_dinamico_participativo,dynamic_part," << chunkSize << ",1\n";
            csvFile.close();
        }
        MPI_Finalize();
        return 0;
    }

    // -------------------------------------------------------------------------
    // PAPEL DO COORDENADOR PARTICIPATIVO (RANK 0)
    // -------------------------------------------------------------------------
    if (rank == 0) {
        std::cout << "========================================================\n"
                  << "  Benchmark Mandelbrot - MPI Dinamico (Coordenador PARTICIPATIVO)\n"
                  << "  Processos Totais : " << size << " (TODOS os " << size << " processos calculam pixels)\n"
                  << "  Granularidade    : " << chunkSize << " linha(s) por tarefa\n"
                  << "  Resolucao        : " << WIDTH << "x" << HEIGHT << " (" << MAX_ITER << " max iter)\n"
                  << "========================================================\n" << std::flush;

        std::vector<int> temps(HEIGHT * WIDTH, 0);

        int next_row = 0;
        int active_workers = 0;

        // 1. Despacha primeira leva de tarefas para os trabalhadores (ranks 1 a size-1)
        for (int worker = 1; worker < size; ++worker) {
            if (next_row < HEIGHT) {
                int rows_to_send = std::min(chunkSize, HEIGHT - next_row);
                int task[2] = {next_row, rows_to_send};
                MPI_Send(task, 2, MPI_INT, worker, TAG_WORK, MPI_COMM_WORLD);
                next_row += rows_to_send;
                active_workers++;
            } else {
                int stop_signal[2] = {-1, 0};
                MPI_Send(stop_signal, 2, MPI_INT, worker, TAG_STOP, MPI_COMM_WORLD);
            }
        }

        std::vector<int> recv_buffer(2 + chunkSize * WIDTH);

        // Função auxiliar interna: atende qualquer trabalhador pronto via MPI_Iprobe
        auto check_and_serve_workers = [&]() {
            int flag = 0;
            MPI_Status status;
            // Esvazia a fila de trabalhadores prontos para evitar starvation
            while (MPI_Iprobe(MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &flag, &status) && flag) {
                int worker = status.MPI_SOURCE;
                MPI_Recv(recv_buffer.data(), recv_buffer.size(), MPI_INT,
                         worker, TAG_RESULT, MPI_COMM_WORLD, &status);

                int start_row = recv_buffer[0];
                int num_rows  = recv_buffer[1];

                // Armazena pixels calculados pelo trabalhador
                for (int r = 0; r < num_rows; ++r) {
                    int target_j = start_row + r;
                    std::copy(recv_buffer.begin() + 2 + r * WIDTH,
                              recv_buffer.begin() + 2 + (r + 1) * WIDTH,
                              temps.begin() + target_j * WIDTH);
                }

                // Despacha nova tarefa para o trabalhador ou sinaliza parada
                if (next_row < HEIGHT) {
                    int rows_to_send = std::min(chunkSize, HEIGHT - next_row);
                    int task[2] = {next_row, rows_to_send};
                    MPI_Send(task, 2, MPI_INT, worker, TAG_WORK, MPI_COMM_WORLD);
                    next_row += rows_to_send;
                } else {
                    int stop_signal[2] = {-1, 0};
                    MPI_Send(stop_signal, 2, MPI_INT, worker, TAG_STOP, MPI_COMM_WORLD);
                    active_workers--;
                }
            }
        };

        // 2. Loop de processamento do Coordenador:
        // Enquanto houver linhas no pool de tarefas, o Coordenador pega fatias para SI MESMO
        double my_compute_time = 0.0;

        while (next_row < HEIGHT) {
            // Pega um lote de trabalho para o próprio Rank 0
            int my_start_row = next_row;
            int my_num_rows  = std::min(chunkSize, HEIGHT - next_row);
            next_row += my_num_rows;

            // Calcula o lote de linhas localmente no Rank 0
            for (int r = 0; r < my_num_rows; ++r) {
                int j = my_start_row + r;
                double y = IM_MAX - j * tamPixel_im;
                double t_c0 = MPI_Wtime();
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
                my_compute_time += (MPI_Wtime() - t_c0);

                // A cada linha calculada, sonda a rede e atende trabalhadores pendentes
                check_and_serve_workers();
            }
        }

        // 3. Quando não há mais tarefas para despachar (next_row == HEIGHT),
        // o Coordenador aguarda bloqueado a conclusão dos trabalhadores que ainda estão ocupados
        while (active_workers > 0) {
            MPI_Status status;
            MPI_Recv(recv_buffer.data(), recv_buffer.size(), MPI_INT,
                     MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &status);

            int worker    = status.MPI_SOURCE;
            int start_row = recv_buffer[0];
            int num_rows  = recv_buffer[1];

            for (int r = 0; r < num_rows; ++r) {
                int target_j = start_row + r;
                std::copy(recv_buffer.begin() + 2 + r * WIDTH,
                          recv_buffer.begin() + 2 + (r + 1) * WIDTH,
                          temps.begin() + target_j * WIDTH);
            }

            // Não há mais trabalho: dispensa o trabalhador
            int stop_signal[2] = {-1, 0};
            MPI_Send(stop_signal, 2, MPI_INT, worker, TAG_STOP, MPI_COMM_WORLD);
            active_workers--;
        }

        // Fim da medição de tempo total da etapa paralela
        double t_end = MPI_Wtime();
        double elapsed = t_end - t_start;

        std::cout << "Tempo gasto (Coordenador Participativo total): " << elapsed << " segundos.\n";

        // Garante diretórios
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

            std::string prefix = "out/mandelbrot_mpi_dinamico_part_chunk" + std::to_string(chunkSize) + "_" + std::to_string(executionNum);

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
                    << ",mpi_dinamico_participativo"
                    << ",dynamic_part"
                    << "," << chunkSize
                    << "," << size
                    << "\n";
            csvFile.close();
            std::cout << "Registro salvo em '" << csvPath << "'\n";
        }

        // Coleta os tempos de computação de todos os processos (inclusive Rank 0)
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
                            << ",mpi_dinamico_participativo,dynamic_part," << chunkSize << "," << size
                            << "," << WIDTH << "," << HEIGHT << "," << MAX_ITER
                            << "," << r << "," << all_proc_times[r] << "\n";
            }
            procCsvFile.close();
        }

        std::cout << "Execucao MPI Dinamica (Coordenador Participativo) finalizada com exito!\n";

    } else {
        // ---------------------------------------------------------------------
        // PAPEL DOS PROCESSOS TRABALHADORES (RANKS 1 A SIZE-1)
        // ---------------------------------------------------------------------
        std::vector<int> send_buffer(2 + chunkSize * WIDTH);
        double my_compute_time = 0.0;

        while (true) {
            int task[2];
            MPI_Status status;
            MPI_Recv(task, 2, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_STOP || task[0] < 0) {
                break;
            }

            int start_row = task[0];
            int num_rows  = task[1];

            send_buffer[0] = start_row;
            send_buffer[1] = num_rows;

            double t_w0 = MPI_Wtime();
            for (int r = 0; r < num_rows; ++r) {
                int j = start_row + r;
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
                    send_buffer[2 + r * WIDTH + i] = temp;
                }
            }
            my_compute_time += (MPI_Wtime() - t_w0);

            MPI_Send(send_buffer.data(), 2 + num_rows * WIDTH, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
        }

        // Envia seu tempo total de computação para o Rank 0
        MPI_Gather(&my_compute_time, 1, MPI_DOUBLE, nullptr, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
