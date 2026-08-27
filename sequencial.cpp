#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <vector>
#include <cstdlib>

class Complex {
public:
    double real;
    double imag;
    const double EPSILON = 1e-9;

    Complex(double r = 0.0, double i = 0.0) : real(r), imag(i) {}

    Complex operator+(const Complex& other) const {
        return Complex(real + other.real, imag + other.imag);
    }

    Complex operator-(const Complex& other) const {
        return Complex(real - other.real, imag - other.imag);
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

    bool operator==(const Complex& other) const {
        return std::abs(real - other.real) < EPSILON && std::abs(imag - other.imag) < EPSILON;
    }

    bool operator==(double val) const {
        return std::abs(real - val) < EPSILON && std::abs(imag) < EPSILON;
    }

    // Retorna o quadrado do módulo (útil para otimização em cálculos de fractais)
    double magnitudeSquared() const {
        return real * real + imag * imag;
    }

    Complex point(double x, double y){
        real = x;
        imag = y;
        return *this;
    }

    void print() const {
        std::cout << real << (imag >= 0 ? "+" : "" ) << imag << "i" << std::endl;
    }
};

int main() {
    // Começa a marcar o tempo
    auto start = std::chrono::high_resolution_clock::now();

    // -------------------------------------------------------------
    // Cálculo
    // -------------------------------------------------------------

    // Parâmetros padrão
    int WIDTH = 4096, HEIGHT = 4096, MAX_ITER = 1000;
    double RE_MIN = -2.0, RE_MAX = 1.0, IM_MIN = -1.5, IM_MAX = 1.5;

    std::ifstream inFile("in.txt");
    if (inFile.is_open()) {
        std::string line;
        while (std::getline(inFile, line)) {
            // Ignora linhas vazias ou comentários (caso queira adicionar no futuro)
            if (line.empty() || line[0] == '#') continue; 
            
            size_t pos = line.find('=');
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 1);
                
                if (key == "WIDTH") WIDTH = std::stoi(value);
                else if (key == "HEIGHT") HEIGHT = std::stoi(value);
                else if (key == "MAX_ITER") MAX_ITER = std::stoi(value);
                else if (key == "RE_MIN") RE_MIN = std::stod(value);
                else if (key == "RE_MAX") RE_MAX = std::stod(value);
                else if (key == "IM_MIN") IM_MIN = std::stod(value);
                else if (key == "IM_MAX") IM_MAX = std::stod(value);
            }
        }
        inFile.close();
    } else {
        std::cerr << "Aviso: Nao foi possivel abrir in.txt. Utilizando resolucao 4096 e params padroes." << std::endl;
    }

    double tamPixel_re = (RE_MAX - RE_MIN)/WIDTH,
           tamPixel_im = (IM_MAX - IM_MIN)/HEIGHT;
    int limHeight = (HEIGHT%2==0)? HEIGHT/2:HEIGHT/2+1;
    
    // Alocamos apenas a metade superior para economizar memória e evitar estouro de pilha
    std::vector<int> temps(limHeight * WIDTH, 0);

    for (int i = 0; i < WIDTH; i++) {
        double x = RE_MIN + i * tamPixel_re;
        for (int j = 0; j < limHeight; j++) {
            double y = IM_MIN + j * tamPixel_im;
            Complex c(x, y);
            Complex z = 0;
            int temp = 0;
            
            // Loop de escape de Mandelbrot
            while (z.magnitudeSquared() < 4.0 && temp < MAX_ITER) {
                z = z * z + c;
                temp++;
            }
            
            temps[j * WIDTH + i] = temp;
        }
    }

    // Termina de marcar o tempo
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Determina o número da execução atual com base nas linhas registradas no CSV
    int executionNum = 1;
    std::ifstream checkFile("dateTimeExecution.csv");
    if (checkFile.is_open()) {
        std::string line;
        while (std::getline(checkFile, line)) {
            if (!line.empty()) {
                executionNum++;
            }
        }
        checkFile.close();
    }

    // Grava a tabela de inteiros para validação
    std::string tableFilename = "mandelbrot_table_" + std::to_string(executionNum) + ".txt";
    std::ofstream tableFile(tableFilename);
    if (tableFile.is_open()) {
        for (int j = 0; j < HEIGHT; ++j) {
            int targetRow = (j < limHeight) ? j : (HEIGHT - 1 - j);
            for (int i = 0; i < WIDTH; ++i) {
                tableFile << temps[targetRow * WIDTH + i] << (i == WIDTH - 1 ? "" : " ");
            }
            tableFile << "\n";
        }
        tableFile.close();
        std::cout << "Tabela de iteracoes salva com sucesso em '" << tableFilename << "'\n";
    } else {
        std::cerr << "Erro ao abrir " << tableFilename << " para escrita.\n";
    }

    // Grava a matriz em formato PPM P3 (Escala de Cinza)
    std::string ppmFilename = "mandelbrot_" + std::to_string(executionNum) + ".ppm";
    std::ofstream ppmFile(ppmFilename);
    if (ppmFile.is_open()) {
        ppmFile << "P3\n" << WIDTH << " " << HEIGHT << "\n255\n";
        for (int j = 0; j < HEIGHT; ++j) {
            int targetRow = (j < limHeight) ? j : (HEIGHT - 1 - j);
            for (int i = 0; i < WIDTH; ++i) {
                int val = temps[targetRow * WIDTH + i];
                if (val >= MAX_ITER) {
                    ppmFile << "0 0 0 "; // Preto (pontos dentro do conjunto)
                } else {
                    // Colorização RGB: três senos com fases e frequências diferentes.
                    // Cada canal oscila em ritmo próprio, criando um gradiente de cores ricas.
                    // A fase (0.0, 2.094, 4.188) = (0, 2π/3, 4π/3) separa R, G e B em 120°
                    // criando uma roda de cores completa que cicla suavemente com as iterações.
                    double t = static_cast<double>(val);
                    int r = static_cast<int>(std::sin(0.016 * t + 0.0)   * 127 + 128);
                    int g = static_cast<int>(std::sin(0.013 * t + 2.094) * 127 + 128);
                    int b = static_cast<int>(std::sin(0.010 * t + 4.188) * 127 + 128);
                    ppmFile << r << " " << g << " " << b << " ";
                }
            }
            ppmFile << "\n";
        }
        ppmFile.close();
        std::cout << "Imagem em escala de cinza salva com sucesso em '" << ppmFilename << "'\n";
    } else {
        std::cerr << "Erro ao abrir " << ppmFilename << " para escrita.\n";
    }

    // Obtém a data e hora atual do sistema
    auto now = std::chrono::system_clock::now();
    std::time_t current_time = std::chrono::system_clock::to_time_t(now);

    // Verifica se o arquivo CSV é novo ou está vazio para gravar o cabeçalho
    bool isNewFile = false;
    {
        std::ifstream testFile("dateTimeExecution.csv");
        if (!testFile || testFile.peek() == std::ifstream::traits_type::eof()) {
            isNewFile = true;
        }
    }

    // Salva o resultado adicionando (append) no arquivo CSV
    std::ofstream csvFile("dateTimeExecution.csv", std::ios::app);
    if (csvFile.is_open()) {
        const char* machineName = std::getenv("COMPUTERNAME");
        if (!machineName) machineName = std::getenv("HOSTNAME");
        if (!machineName) machineName = "Unknown";

        if (isNewFile) {
            csvFile << "DataHora,TempoGasto,WIDTH,HEIGHT,MAX_ITER,RE_MIN,RE_MAX,IM_MIN,IM_MAX,Machine\n";
        }
        
        // Formato: YYYY-MM-DD HH:MM:SS,TempoGasto,WIDTH,HEIGHT,MAX_ITER,RE_MIN,RE_MAX,IM_MIN,IM_MAX,Machine
        csvFile << std::put_time(std::localtime(&current_time), "%Y-%m-%d %H:%M:%S") 
                << "," << elapsed.count()
                << "," << WIDTH
                << "," << HEIGHT
                << "," << MAX_ITER
                << "," << RE_MIN
                << "," << RE_MAX
                << "," << IM_MIN
                << "," << IM_MAX 
                << "," << machineName << "\n";
        csvFile.close();
        
        std::cout << "Execução finalizada!\n";
        std::cout << "Tempo gasto: " << elapsed.count() << " segundos.\n";
        std::cout << "Registro salvo em 'dateTimeExecution.csv'\n";
    } else {
        std::cerr << "Erro ao abrir dateTimeExecution.csv para escrita.\n";
    }

    return 0;
}
