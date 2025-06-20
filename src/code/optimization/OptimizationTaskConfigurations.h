//
// Created by Yifei Li on 3/3/21.
// Email: liyifei@csail.mit.edu
//

#ifndef OMEGAENGINE_SIMULATIONCONSTANTS_H
#define OMEGAENGINE_SIMULATIONCONSTANTS_H

#include "../simulation/Simulation.h"

class OptimizationTaskConfigurations {
public:
    static Simulation::FabricConfiguration normalFabric6lowres, slopeFabricRestOnPlane, conitnuousNormalTestFabric, 
            tshirt1000, tie, towel, agenthat579, sock482, dressv7khandsUpDrape, sphereFabric, normalFabric6,
            mpmClothFabric;

    static Simulation::SceneConfiguration simpleScene, rotatingSphereScene, windScene, tshirtScene,
            hatScene, sockScene, dressScene, continousNormalScene, slopeSimplifiedScene, hangerScene, towelScene,
            mpmClothScene;

    static Simulation::TaskConfiguration demoSphere, demoTshirt, demoWInd, demoHat, demoSock, demoDress, 
            demoWindSim2Real, demoSlope, demoHanger,
            demoMPMCloth;
            
    static std::vector<Simulation::SceneConfiguration> sceneConfigArrays;

    static std::map<int, Simulation::TaskConfiguration> demoNumToConfigMap;

};


#endif //OMEGAENGINE_SIMULATIONCONSTANTS_H