//#include "TcpClient.h"
//
//TcpClient::TcpClient(QObject* parent) : QObject(parent), socket(new QTcpSocket(this))
//{
//    connect(socket, &QTcpSocket::connected, this, &TcpClient::onConnected);
//    connect(socket, &QTcpSocket::disconnected, this, &TcpClient::onDisconnected);
//    connect(socket, &QTcpSocket::readyRead, this, &TcpClient::onReadyRead);
//}
//
//void TcpClient::connectToServer(const QString& host, quint16 port)
//{
//    socket->connectToHost(host, port);
//}
//
//void TcpClient::onConnected()
//{
//    qDebug() << "Connected to server.";
//    socket->write("Hello from the client!");
//}
//
//void TcpClient::onDisconnected()
//{
//    qDebug() << "Disconnected from server.";
//}
//
//void TcpClient::onReadyRead()
//{
//    QByteArray data = socket->readAll();
//    qDebug() << "Received from server:" << data;
//}
