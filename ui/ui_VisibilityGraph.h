/********************************************************************************
** Form generated from reading UI file 'VisibilityGraph.ui'
**
** Created by: Qt User Interface Compiler version 5.13.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_VISIBILITYGRAPH_H
#define UI_VISIBILITYGRAPH_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_FlowEncoderClass
{
public:
    QWidget *centralWidget;

    void setupUi(QMainWindow *FlowEncoderClass)
    {
        if (FlowEncoderClass->objectName().isEmpty())
            FlowEncoderClass->setObjectName(QString::fromUtf8("FlowEncoderClass"));
        FlowEncoderClass->resize(521, 522);
        centralWidget = new QWidget(FlowEncoderClass);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        FlowEncoderClass->setCentralWidget(centralWidget);

        retranslateUi(FlowEncoderClass);

        QMetaObject::connectSlotsByName(FlowEncoderClass);
    } // setupUi

    void retranslateUi(QMainWindow *FlowEncoderClass)
    {
        FlowEncoderClass->setWindowTitle(QCoreApplication::translate("FlowEncoderClass", "FlowEncoder", nullptr));
    } // retranslateUi

};

namespace Ui {
    class FlowEncoderClass: public Ui_FlowEncoderClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_VISIBILITYGRAPH_H
