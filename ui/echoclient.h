#ifndef ECHOCLIENT_H
#define ECHOCLIENT_H

#include <QtCore/QObject>
#include <QtCore/QDebug>
#include <QtWebSockets/QWebSocket>

QT_USE_NAMESPACE

class EchoClient : public QObject {
    Q_OBJECT
public:
    explicit EchoClient(const QUrl& url, bool debug = false, QObject* parent = nullptr)
        : QObject(parent), m_url(url), m_debug(debug) {
        if (m_debug)
            qDebug() << "WebSocket server:" << url;
        connect(&m_webSocket, &QWebSocket::connected, this, &EchoClient::onConnected);
        connect(&m_webSocket, &QWebSocket::disconnected, this, &EchoClient::closed);
        m_webSocket.open(QUrl(url));
    }

    void onConnected() {
        if (m_debug)
            qDebug() << "WebSocket connected";
        connect(&m_webSocket, &QWebSocket::textMessageReceived, this, &EchoClient::onTextMessageReceived);
        m_webSocket.sendTextMessage(QStringLiteral("Hello, world!"));
    }

    void onTextMessageReceived(QString message) {
        if (m_debug)
            qDebug() << "Message received:" << message;
        m_webSocket.close();
    }

Q_SIGNALS:
    void closed();

private:
    QWebSocket m_webSocket;
    QUrl m_url;
    bool m_debug;
};

#endif // ECHOCLIENT_H
