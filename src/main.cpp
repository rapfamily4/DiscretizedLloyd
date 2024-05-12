#include "application.hpp"

int main(int argc, char* argv[]) {
    Application app;
    if (app.init()) {
        app.printUsage();
        app.launch();
    }

    return 0;
}
