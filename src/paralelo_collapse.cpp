#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <vector>
#include <cstdlib>
#include <filesystem>
#include <omp.h>

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

int main() {
    // Parâmetros padrão
    int WIDTH = 4096, HEIGHT = 4096, MAX_ITER = 1000;
    double RE_MIN = -2.0, RE_MAX = 1.0, IM_MIN = -1.5, IM_MAX = 1.5;

    // Parâmetros de escalonamento OpenMP
    std::string schedName = "dynamic"; // static | dynamic | guided | auto
    int chunkSize = 0;                 // 0 = deixa o OpenMP escolher o padrão
    int numThreads = 0;                // 0 = padrão do ambiente / sistema

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
                else if (key == "THREADS")    numThreads = std::stoi(value);
            }
        }
        inFile.close();
    } else {
        std::cerr << "Aviso: Nao foi possivel abrir in.txt. Utilizando resolucao 4096 e params padroes." << std::endl;
    }

    if (numThreads > 0) {
        omp_set_num_threads(numThreads);
    }

    // Converte o nome do escalonamento para o tipo OpenMP e aplica em runtime
    omp_sched_t schedKind = omp_sched_dynamic;
    if      (schedName == "static")  schedKind = omp_sched_static;
    else if (schedName == "dynamic") schedKind = omp_sched_dynamic;
    else if (schedName == "guided")  schedKind = omp_sched_guided;
    else if (schedName == "auto")    schedKind = omp_sched_auto;
    else {
        std::cerr << "Aviso: SCHEDULE='" << schedName << "' desconhecido. Usando 'dynamic'.\n";
        schedName = "dynamic";
    }
    omp_set_schedule(schedKind, chunkSize); // define o escalonamento em runtime

    double tamPixel_re = (RE_MAX - RE_MIN) / WIDTH,
           tamPixel_im = (IM_MAX - IM_MIN) / HEIGHT;

    // Garante caminho padronizado da base de dados com fallback
    std::string csvPath = "data/dateTimeExecution.csv";
    if (!std::filesystem::exists("data/dateTimeExecution.csv") && std::filesystem::exists("dateTimeExecution.csv")) {
        csvPath = "dateTimeExecution.csv";
    } else {
        std::filesystem::create_directories("data");
    }

    // Determina o número da execução atual com base nas linhas registradas no CSV
    int executionNum = 1;
    std::ifstream checkFile(csvPath);
    if (checkFile.is_open()) {
        std::string line;
        while (std::getline(checkFile, line)) {
            if (!line.empty()) executionNum++;
        }
        checkFile.close();
    }

    // Aloca a matriz completa para toda a imagem
    std::vector<int> temps(HEIGHT * WIDTH, 0);

    std::cout << "Iniciando [collapse(2)] com schedule=" << schedName
              << " chunk=" << chunkSize
              << " threads=" << omp_get_max_threads() << "\n";

    // Começa a marcar o tempo
    auto start = std::chrono::high_resolution_clock::now();

    // Loop paralelo com collapse(2): funde os dois loops em um único espaço
    // de iteração de HEIGHT*WIDTH pixels, distribuindo pixels individuais
    // entre as threads. Chunk size agora é em pixels, não em linhas.
    #pragma omp parallel for collapse(2) schedule(runtime) shared(temps)
    for (int j = 0; j < HEIGHT; ++j) {
        for (int i = 0; i < WIDTH; ++i) {
            double y = IM_MAX - j * tamPixel_im;
            double x = RE_MIN + i * tamPixel_re;
            Complex c(x, y);
            Complex z = 0;
            int temp = 0;

            // Loop de escape de Mandelbrot
            while (z.magnitudeSquared() < 4.0 && temp < MAX_ITER) {
                z = z * z + c;
                temp++;
            }

            // Grava na matriz (acesso a índices distintos, sem race condition)
            temps[j * WIDTH + i] = temp;
        }
    }

    // Termina de marcar o tempo
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Garante que a pasta out exista
    std::filesystem::create_directories("out");

    // Prefixo dos arquivos inclui o modo de escalonamento
    std::string prefix = "out/mandelbrot_col_" + schedName + "_" + std::to_string(executionNum);

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
        std::cout << "Tabela de iteracoes salva com sucesso em '" << tableFilename << "'\n";
    } else {
        std::cerr << "Erro ao abrir " << tableFilename << " para escrita.\n";
    }

    std::string ppmFilename = prefix + "_img.ppm";
    std::ofstream ppmFile(ppmFilename);
    if (ppmFile.is_open()) {
        int blackLineWidth = static_cast<int>(WIDTH * 0.01); // Linha preta 1% da largura
        int gradientWidth  = static_cast<int>(WIDTH * 0.10); // Gradiente 10% da largura
        int ppmWidth = WIDTH + blackLineWidth + gradientWidth;

        ppmFile << "P3\n" << ppmWidth << " " << HEIGHT << "\n255\n";
        for (int j = 0; j < HEIGHT; ++j) {
            for (int i = 0; i < WIDTH; ++i) {
                int temp = temps[j * WIDTH + i];
                if (temp >= MAX_ITER) {
                    ppmFile << "0 0 0 "; // Preto
                } else {
                    RGB color = getPaletteColor(static_cast<double>(temp) / MAX_ITER);
                    ppmFile << color.r << " " << color.g << " " << color.b << " ";
                }
            }

            // Finaliza a linha no PPM (com linha preta e gradiente)
            for (int i = 0; i < blackLineWidth; ++i) {
                ppmFile << "0 0 0 ";
            }

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

    // Obtém a data e hora atual do sistema
    auto now = std::chrono::system_clock::now();
    std::time_t current_time = std::chrono::system_clock::to_time_t(now);

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

        csvFile << std::put_time(std::localtime(&current_time), "%Y-%m-%d %H:%M:%S")
                << "," << elapsed.count()
                << "," << WIDTH
                << "," << HEIGHT
                << "," << MAX_ITER
                << "," << RE_MIN
                << "," << RE_MAX
                << "," << IM_MIN
                << "," << IM_MAX
                << "," << machineName
                << ",paralelo_collapse"
                << "," << schedName
                << "," << chunkSize
                << "," << omp_get_max_threads()
                << "\n";
        csvFile.close();

        std::cout << "Execucao Paralela Collapse(2) (OpenMP) finalizada!\n";
        std::cout << "Schedule: " << schedName << "  Chunk: " << chunkSize
                  << "  Threads: " << omp_get_max_threads() << "\n";
        std::cout << "Tempo gasto (puro processamento): " << elapsed.count() << " segundos.\n";
        std::cout << "Registro salvo em '" << csvPath << "'\n";
    }

    return 0;
}
