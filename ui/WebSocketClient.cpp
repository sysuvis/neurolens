// WebSocketClient.cpp
#include "WebSocketClient.h"
#include <iostream>

WebSocketClient::WebSocketClient(const QUrl& url, QObject* parent) : QObject(parent), m_url(url)
{
    connect(&m_webSocket, &QWebSocket::connected, this, &WebSocketClient::onConnected);
    connect(&m_webSocket, &QWebSocket::textMessageReceived,
        this, &WebSocketClient::onTextMessageReceived);
    connect(&m_webSocket, &QWebSocket::disconnected, this, &WebSocketClient::closed);
}

void WebSocketClient::connectToServer()
{
    m_webSocket.open(m_url);
}

void WebSocketClient::onConnected()
{
    std::cout << "WebSocket connected" << std::endl;
    m_webSocket.sendTextMessage(QStringLiteral("Hello, WebSocket Server!"));
}

void WebSocketClient::onTextMessageReceived(const QString& message)
{
    std::cout << "Message received: " << message.toStdString() << std::endl;
    m_webSocket.close();
}
