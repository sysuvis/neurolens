// Communication.h
#ifndef COMMUNICATION_H
#define COMMUNICATION_H
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <string>

//#pragma comment(lib, "Ws2_32.lib")

class Communication {
public:
    static std::string sendPostRequest(const std::string& requestData, const std::string& url = "127.0.0.1", const std::string& port = "5000") {
        WSADATA wsaData;
        WSAStartup(MAKEWORD(2, 2), &wsaData);
        SOCKET sock = createSocket(url, stoi(port));

        std::string requestHeader =
            "POST /process HTTP/1.1\r\n"
            "Host: " + url + ":" + port + "\r\n"
            "Content-Type: text/plain\r\n"
            "Content-Length: " + std::to_string(requestData.length()) + "\r\n"
            "Connection: close\r\n\r\n";

        std::string request = requestHeader + requestData;

        send(sock, request.c_str(), request.length(), 0);

        std::string response = receiveResponse(sock);
        closesocket(sock);
        WSACleanup();
        return response;
    }

    static std::string sendGetRequest(const std::string& url, const std::string& port, const std::string& message) {
        WSADATA wsaData;
        WSAStartup(MAKEWORD(2, 2), &wsaData);
        SOCKET sock = createSocket(url, stoi(port));

        std::string request =
            "GET /get_response?message=" + message + " HTTP/1.1\r\n"
            "Host: " + url + ":" + port + "\r\n"
            "Connection: close\r\n\r\n";

        send(sock, request.c_str(), request.length(), 0);

        std::string response = receiveResponse(sock);
        closesocket(sock);
        WSACleanup();
        return response;
    }

    static std::string extractResponseBody(const std::string& response) {
        auto headerEnd = response.find("\r\n\r\n");
        if (headerEnd != std::string::npos) {
            return response.substr(headerEnd + 4); // Skip the header
        }
        return ""; // 如果没有找到头部，返回空字符串
    }


private:
    static SOCKET createSocket(const std::string& url, int port) {
        sockaddr_in serverAddr;
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_port = htons(port);
        inet_pton(AF_INET, url.c_str(), &serverAddr.sin_addr);

        SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);
        connect(sock, (sockaddr*)&serverAddr, sizeof(serverAddr));
        return sock;
    }

    static std::string receiveResponse(SOCKET sock) {
        std::string response;
        const int bufferSize = 512;
        char buffer[bufferSize];

        int bytesReceived = 0;
        do {
            bytesReceived = recv(sock, buffer, bufferSize - 1, 0);
            if (bytesReceived > 0) {
                buffer[bytesReceived] = '\0'; // 确保字符串以空字符结尾
                response += buffer;
            }
        } while (bytesReceived > 0);

        if (bytesReceived < 0) {
            // 处理接收错误...
        }

        return response;
    }
};

#endif // COMMUNICATION_H