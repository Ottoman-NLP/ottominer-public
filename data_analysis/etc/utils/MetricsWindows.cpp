#include <QApplication>
#include <QProgressBar>
#include <QLabel>
#include <QVBoxLayout>
#include <QWidget>
#include <Pybind11/pybind11.h>
#include <pybind11/embed.h>


namespace py = pybind11;

class MetricsWindow : public QWidget {
    Q_OBJECT

public:
    MetricsWindow(QWidget *parent = nullptr): QWidget(parent) {

        QVBoxLayout *layout = new QVBoxLayout;
        
        progressBar = new QProgressBar(this);
        progressBar->setRange(0, 100);
        layout->addWidget(progressBar);

        cpulabel = new QLabel("CPU Usage: %0", this);
        layout->addWidget(cpulabel);

        gpuLabel = new QLabel("GPU Usage: %0", this);
        layout->addWidget(gpuLabel);

    }

    void updateProgress(int value) {
        progressBar->setValue(value);
    }

    void updateCpuUsage(float value) {
        cpuLabel->setText(QString("CPU Usage: %1%").arg(value));

    void updateGpuUsage(float value) {
        gpuLabel->setText(QString("GPU Usage: %1%").arg(value));

    }

private:
    QProgressBar *progressBar;
    QLabel *cpuLabel;
    QLabel *gpuLabel;
};

    PYBIND11_EMBEDDED_MODULE(MetricsWindow, m) {
        py::class_<MetricsWindow>(m, "MetricsWindow")
            .def(py::init<>())
            .def("updateProgress", &MetricsWindow::updateProgress)
            .def("updateCpuUsage", &MetricsWindow::updateCpuUsage)
            .def("updateGpuUsage", &MetricsWindow::updateGpuUsage);


}

#include "MetricsWindow.moc"
