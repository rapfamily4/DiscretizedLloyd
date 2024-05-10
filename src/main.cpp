#include "application.hpp"

int main(int argc, char* argv[]) {
/*
    // Load mesh from file.
    std::string modelPath = argc > 1 ? std::string(argv[1]) : "./models/plane.obj";
    if (!viewer.load_mesh_from_file(modelPath)) return -1;
    model = Model(viewer.data().V, viewer.data().F, viewer.data().V_normals);

 */ 
    Application app;
    if (app.init()) {
        app.printUsage();
        app.launch();
    }

    return 0;
}
