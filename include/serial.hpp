#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>

class SerialPort {
public:
    SerialPort(const char* port, int baudrate) : fd_(-1) {
        fd_ = open(port, O_RDWR | O_NOCTTY | O_NDELAY);
        if (fd_ == -1) {
            throw std::runtime_error("Unable to open serial port");
        }

        struct termios options{};
        tcgetattr(fd_, &options);
        cfsetispeed(&options, baudrate);
        cfsetospeed(&options, baudrate);

        options.c_cflag |= (CLOCAL | CREAD);
        options.c_cflag &= ~PARENB;
        options.c_cflag &= ~CSTOPB;
        options.c_cflag &= ~CSIZE;
        options.c_cflag |= CS8;
        options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
        options.c_oflag &= ~OPOST;

        options.c_cc[VMIN] = 1;
        options.c_cc[VTIME] = 0;

        tcsetattr(fd_, TCSANOW, &options);
    }

    ~SerialPort() {
        if (fd_ != -1) close(fd_);
    }

    bool isOpen() const { return fd_ != -1; }

    void write(const char* data) const {
        ::write(fd_, data, strlen(data));
    }

    std::string readUntil(char delim = 'b') {
        std::string result;
        char ch;
        while (true) {
            ssize_t n = ::read(fd_, &ch, 1);
            if (n == 1) {
                result += ch;
                if (ch == delim) break;
            }
        }
        return result;
    }

private:
    int fd_;
};