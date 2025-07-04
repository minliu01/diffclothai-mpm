#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "simulation/Simulation.h"
#include "engine/Constants.h"
#include "optimization/OptimizationTaskConfigurations.h"
#include "engine/RenderLoop.h"
#include "engine/Debug.h"

namespace py = pybind11;

std::map<std::string, std::string> convertDictToSceneConfig(py::dict dictionary) {
    // added by Min for MPM_CLOTH
    std::map<std::string, std::string> result;
    for (std::pair<py::handle, py::handle> item : dictionary) {
        auto key = item.first.cast<std::string>();
        auto value = item.second.cast<std::string>();
        result[key] = value;
    }
    return result;
}

void customize_scene_from_config(Simulation::SceneConfiguration &sceneConfig, const std::map<std::string, std::string> config) {
    // added by Min for MPM_CLOTH
    // fabric
    if (config.find("fabric:k_stiff_stretching") != config.end())
      sceneConfig.fabric.k_stiff_stretching = std::stod(config.at("fabric:k_stiff_stretching"));
    if (config.find("fabric:k_stiff_bending") != config.end())
      sceneConfig.fabric.k_stiff_bending = std::stod(config.at("fabric:k_stiff_bending"));
    if (config.find("fabric:keepOriginalScalePoint") != config.end())
      sceneConfig.fabric.keepOriginalScalePoint = (config.at("fabric:keepOriginalScalePoint") == "true");
    if (config.find("fabric:custominitPos") != config.end()){
      sceneConfig.fabric.custominitPos = (config.at("fabric:custominitPos") == "true");
      if (sceneConfig.fabric.custominitPos)
        sceneConfig.fabric.initPosFile = config.at("fabric:initPosFile");
    }
    if (config.find("fabric:name") != config.end())
      sceneConfig.fabric.name = config.at("fabric:name");
    if (config.find("fabric:density") != config.end())
      sceneConfig.fabric.density = std::stod(config.at("fabric:density"));
    // scene
    if (config.find("timeStep") != config.end())
      sceneConfig.timeStep = std::stod(config.at("timeStep"));
    if (config.find("stepNum") != config.end())
      sceneConfig.stepNum = std::stoi(config.at("stepNum"));
    if (config.find("forwardConvergenceThresh") != config.end())
      sceneConfig.forwardConvergenceThresh = std::stod(config.at("forwardConvergenceThresh"));
    if (config.find("backwardConvergenceThresh") != config.end())
      sceneConfig.backwardConvergenceThresh = std::stod(config.at("backwardConvergenceThresh"));
    if (config.find("attachmentPoints") != config.end() && config.at("attachmentPoints") == "CUSTOM_ARRAY") {
      sceneConfig.attachmentPoints = AttachmentConfigs::CUSTOM_ARRAY;
      sceneConfig.customAttachmentVertexIdx = {{0.0, {}}};
      std::stringstream ss(config.at("customAttachmentVertexIdx"));
      while (ss.good()) {
        std::string substr;
        getline(ss, substr, ',');
        sceneConfig.customAttachmentVertexIdx[0].second.push_back(std::stoi(substr));
      }
    }
    if (config.find("orientation") != config.end()) {
      if (config.at("orientation") == "FRONT")
        sceneConfig.orientation = Orientation::FRONT;
      else if (config.at("orientation") == "DOWN")
        sceneConfig.orientation = Orientation::DOWN;
      else if (config.at("orientation") == "BACK")
        sceneConfig.orientation = Orientation::BACK;
      else if (config.at("orientation") == "CUSTOM_ORIENTATION"){
        sceneConfig.orientation = Orientation::CUSTOM_ORIENTATION;
        sceneConfig.upVector = Vec3d(0,0,0);
        std::stringstream ss(config.at("upVector"));
        int idx = 0;
        while (ss.good() && idx < 3) {
          std::string substr;
          getline(ss, substr, ',');
          sceneConfig.upVector[idx++] = std::stod(substr);
        }
      }
    }
    return;
}

Simulation* makeCustomizedSim(std::string exampleName, bool runBackward = true, py::dict config = py::dict()) {
  // added by Min for MPM_CLOTH
  Simulation::forwardConvergenceThreshold = 1e-5;
  Simulation* sim = nullptr;
  std::map<std::string, std::string> custom_config = convertDictToSceneConfig(config);
  if (exampleName == "mpm_cloth") {
    Simulation::SceneConfiguration initSceneProfile = OptimizationTaskConfigurations::mpmClothScene;
    customize_scene_from_config(initSceneProfile, custom_config);
    sim = Simulation::createSystem(initSceneProfile, Vec3d(0, 0, 0), runBackward);
    Vec3d translation = Vec3d(0.8,-1.5,0.);
    sim->taskLossInfo.targetTranslation = translation;
  }
  else {
    throwError("Undefined example name (" + exampleName + ").");
  }
  return sim;
}

Simulation* makeSim(std::string exampleName, bool runBackward = true) {
  Simulation::forwardConvergenceThreshold = 1e-5;
  Simulation* sim = nullptr;
  if (exampleName == "wear_hat") {
    // create simulation instance
    Simulation::SceneConfiguration initSceneProfile = OptimizationTaskConfigurations::hatScene;
    sim = Simulation::createSystem(initSceneProfile, Vec3d(0, 0, 0), runBackward);
    // define loss
    Vec3d bustCenter = sim->sphere_head.center + Vec3d(0, sim->sphere_head.radius * 0.6, 0);
    Vec3d hatCenter = (sim->restShapeMinDim + sim->restShapeMaxDim) * 0.5;
    Vec3d translation = bustCenter - hatCenter;
    sim->taskLossInfo.targetTranslation = translation;
  } else if (exampleName == "hang_cloth") {
    // create simulation instance
    Simulation::SceneConfiguration initSceneProfile = OptimizationTaskConfigurations::towelScene;
    sim = Simulation::createSystem(initSceneProfile, Vec3d(0, 0, 0), runBackward);
    // define loss
    Vec3d translation = Vec3d(0.8,-1.5,0.);
    sim->taskLossInfo.targetTranslation = translation;
  } else if (exampleName == "tie") {
    // create simulation instance
    Simulation::SceneConfiguration initSceneProfile = OptimizationTaskConfigurations::hangerScene;
    sim = Simulation::createSystem(initSceneProfile, Vec3d(0, 0, 0), runBackward);
    // define loss
    Vec3d translation = Vec3d(0.8,-1.5,0.);
    sim->taskLossInfo.targetTranslation = translation;
  } else if (exampleName == "wear_sock") {
    // create simulation instance
    Simulation::SceneConfiguration initSceneProfile = OptimizationTaskConfigurations::sockScene;
    sim = Simulation::createSystem(
                                   initSceneProfile,
                                   Vec3d(0, 0, 0), runBackward);
    // define loss
    Capsule& foot = *(sim->sockLeg.foot);
    Capsule& leg = *(sim->sockLeg.leg);
    Rotation& legRotation =  leg.rotationFromParent;
    Rotation& footRotation = foot.rotationFromParent;
    Vec3d legBaseCenter = sim->sockLeg.center + leg.center ;
    Vec3d footBaseCenter = sim->sockLeg.center + foot.center ;

    Vec3d centerTop = leg.getTransformedPosFromJointBindPos(Vec3d(0, leg.length, 0 ));
    Vec3d centerTopLeft = leg.getTransformedPosFromJointBindPos(Vec3d(-leg.radius, leg.length, 0 ));
    Vec3d centerTopRight = leg.getTransformedPosFromJointBindPos(Vec3d(leg.radius, leg.length, 0 ));
    Vec3d centerTopFront = leg.getTransformedPosFromJointBindPos(Vec3d(0, leg.length, leg.radius));
    Vec3d centerTopBack = leg.getTransformedPosFromJointBindPos(Vec3d(0, leg.length, -leg.radius));

    Vec3d calfPoint =  leg.getTransformedPosFromJointBindPos(Vec3d(0, sim->sockLeg.leg->length * 0.4, -leg.radius));

    Vec3d heelPoint =  foot.getTransformedPosFromJointBindPos(Vec3d(0.0, foot.length, -foot.radius));
    Vec3d archPoint = foot.getTransformedPosFromJointBindPos(Vec3d(0.0, foot.length * 0.5, foot.radius));
    Vec3d toePoint = foot.getTransformedPosFromJointBindPos(Vec3d(0,-foot.radius,0));
    // Vec3d toePoint = foot.getTransformedPosFromJointBindPos(Vec3d(0, 0, foot.radius));
    Vec3d footTipBackPoint = foot.getTransformedPosFromJointBindPos(Vec3d(0.0, 0, -foot.radius));
    Vec3d footTipLeftPoint = foot.getTransformedPosFromJointBindPos(Vec3d(-foot.radius, 0  , 0));
    Vec3d footTipRightPoint = foot.getTransformedPosFromJointBindPos(Vec3d(foot.radius, 0  , 0));
    std::vector<int> topFrontPoints = {104, 27, 43, 475, 392, 903, 416, 413, 895},
            topLeftPoints = {11, 30, 164, 755, 30},
            topRightPoints = {  563, 43, 474, 14},
            toePoints = {865, 420, 946, 250, 80},
            openingBackPoints = {102, 81, 842, 318, 12};

    std::vector<Simulation::CorresPondenceTargetInfo> mappingPairs;
    int lastFrameIdx = OptimizationTaskConfigurations::sockScene.stepNum;
    // add target pairs for stage 1
    mappingPairs.emplace_back(lastFrameIdx, heelPoint, std::vector<int>{2, 20, 336, 792, 995});
    mappingPairs.emplace_back(lastFrameIdx, toePoint, toePoints);
    mappingPairs.emplace_back(lastFrameIdx, archPoint, std::vector<int>{282, 343, 249});
    mappingPairs.emplace_back(lastFrameIdx, centerTopFront, topFrontPoints);
    mappingPairs.emplace_back(lastFrameIdx, centerTopLeft, topLeftPoints); //sock right
    mappingPairs.emplace_back(lastFrameIdx, centerTopRight, topRightPoints); //sock left
    mappingPairs.emplace_back(lastFrameIdx, centerTopBack, openingBackPoints);

    mappingPairs.emplace_back(lastFrameIdx, calfPoint, std::vector<int>{37, 241, 349}); //sock left

    // add target pairs for stage 0
    mappingPairs.emplace_back(0, toePoint, topFrontPoints); //sock left
    mappingPairs.emplace_back(0, footTipBackPoint, openingBackPoints);
    mappingPairs.emplace_back(0, footTipLeftPoint, topLeftPoints);
    mappingPairs.emplace_back(0, footTipRightPoint, topRightPoints);

    // std::cerr << "radius = " << foot.radius << ", " << leg.radius << std::endl;
    sim->taskLossInfo.targetPosPairs = mappingPairs;
    sim->debugPointTargetPos = mappingPairs;
  }
  else {
    throwError("Undefined example name (" + exampleName + ").");
  }
  return sim;
}

OptimizeHelper* makeOptimizeHelperWithSim(std::string exampleName, Simulation* sim) {
  Simulation::forwardConvergenceThreshold = 1e-5;
  OptimizeHelper* helper = nullptr;
  std::cerr << "example Name: " << exampleName << std::endl;
  if (exampleName == "wear_hat") {
    sim->setPrintVerbose(false);
    helper = BackwardTaskSolver::getOptimizeHelperPointer(sim, Demos::DEMO_WEAR_HAT);
  } else if (exampleName == "hang_cloth") {
    sim->setPrintVerbose(false);
    helper = BackwardTaskSolver::getOptimizeHelperPointer(sim, Demos::DEMO_CLOTH_HANGER);
  } else if (exampleName == "tie") {
    sim->setPrintVerbose(false);
    helper = BackwardTaskSolver::getOptimizeHelperPointer(sim, Demos::DEMO_CLOTH_HANGER);
  } else if (exampleName == "mpm_cloth") {
    sim->setPrintVerbose(false);
    helper = BackwardTaskSolver::getOptimizeHelperPointer(sim, Demos::DEMO_MPM_CLOTH);
  } else if (exampleName == "wear_sock") {
    sim->setPrintVerbose(false);
    helper = BackwardTaskSolver::getOptimizeHelperPointer(sim, Demos::DEMO_WEAR_SOCK);
  } else if (exampleName == "wind_tshirt") {
    sim->setPrintVerbose(false);
    Simulation::forwardConvergenceThreshold = 1e-5;
    helper = BackwardTaskSolver::getOptimizeHelperPointer(sim, Demos::DEMO_WIND_TSHIRT);
  } else if (exampleName == "inverse_design") {
    sim->setPrintVerbose(false);
    Simulation::forwardConvergenceThreshold = 1e-6;
    helper = BackwardTaskSolver::getOptimizeHelperPointer(sim, Demos::DEMO_DRESS_TWIRL);
  } else if (exampleName == "wind_sim2real") {
    sim->setPrintVerbose(false);
    Simulation::forwardConvergenceThreshold = 1e-5;
    helper = BackwardTaskSolver::getOptimizeHelperPointer(sim, Demos::DEMO_WIND_SIM2REAL);
  } else if (exampleName == "sphere") {
    sim->setPrintVerbose(false);
    Simulation::forwardConvergenceThreshold = 1e-5;
    helper = BackwardTaskSolver::getOptimizeHelperPointer(sim, Demos::DEMO_SPHERE_ROTATE);
  }
  else {
    throwError("Undefined example name (" + exampleName + ").");
    Simulation::SceneConfiguration initSceneProfile = OptimizationTaskConfigurations::hatScene;
    sim = Simulation::createSystem(
                                   initSceneProfile,
                                   Vec3d(0, 0, 0), false);
    sim->setPrintVerbose(false);
    helper = BackwardTaskSolver::getOptimizeHelperPointer(sim, Demos::DEMO_WEAR_HAT);
  }
  return helper;
}

OptimizeHelper* makeOptimizeHelper(std::string exampleName) {
  Simulation::SceneConfiguration initSceneProfile = OptimizationTaskConfigurations::hatScene;
  Simulation* sim  = Simulation::createSystem(
                                              initSceneProfile,
                                              Vec3d(0, 0, 0), false);
  return makeOptimizeHelperWithSim(exampleName, sim);
}

void enableOpenMP(int n_threads = 5) {
  bool parallelizeEigen = true;

  if (OPENMP_ENABLED) {
    omp_set_num_threads(n_threads);
    if (parallelizeEigen) {
      Eigen::setNbThreads(n_threads);
    }
    // testOmp();
    int n = Eigen::nbThreads();
    std::printf("eigen threads: %d\n", n);
  }
}

void render(Simulation* sim, bool renderPosPairs = false, bool autoExit = false) {
  RenderLoop::renderRecordsForSystem(sim, sim->forwardRecords, renderPosPairs, autoExit, "visualization");
}

PYBIND11_MODULE(diffcloth_py, m) {
  // Primitive
  py::enum_<WindConfig>(m, "WindConfig")
          .value("NO_WIND", WindConfig::NO_WIND)
          .value("WIND_CONSTANT", WindConfig::WIND_CONSTANT)
          .value("WIND_SIN", WindConfig::WIND_SIN)
          .value("WIND_SIN_AND_FALLOFF", WindConfig::WIND_SIN_AND_FALLOFF)
          .value("WIND_FACTOR_PER_STEP", WindConfig::WIND_FACTOR_PER_STEP)
          .export_values();

  // Simulation::SceneConfiguration
  py::class_<Simulation::SceneConfiguration>(m, "SceneConfiguration")
          .def_readwrite("timeStep", &Simulation::SceneConfiguration::timeStep)
          .def_readwrite("windConfig", &Simulation::SceneConfiguration::windConfig)
          .def_readonly("stepNum", &Simulation::SceneConfiguration::stepNum)
          .def_readwrite("customAttachmentVertexIdx", &Simulation::SceneConfiguration::customAttachmentVertexIdx);

  // Simulation::PrimitiveCollisionInformation
  py::class_<Simulation::PrimitiveCollisionInformation>(m, "PrimitiveCollisionInformation")
          .def_readonly("primitiveId", &Simulation::PrimitiveCollisionInformation::primitiveId)
          .def_readonly("particleId", &Simulation::PrimitiveCollisionInformation::particleId)
          .def_readonly("normal", &Simulation::PrimitiveCollisionInformation::normal)
          .def_readonly("v_out", &Simulation::PrimitiveCollisionInformation::v_out)
          .def_readonly("contact_point", &Simulation::PrimitiveCollisionInformation::contact_point)
          .def_readonly("r", &Simulation::PrimitiveCollisionInformation::r)
          .def_readonly("d", &Simulation::PrimitiveCollisionInformation::d)
          .def_readonly("type", &Simulation::PrimitiveCollisionInformation::type);

  // Simulation::SelfCollisionInformation
  py::class_<Simulation::SelfCollisionInformation>(m, "SelfCollisionInformation")
          .def_readonly("particleId1", &Simulation::SelfCollisionInformation::particleId1)
          .def_readonly("particleId2", &Simulation::SelfCollisionInformation::particleId2)
          .def_readonly("d", &Simulation::SelfCollisionInformation::d)
          .def_readonly("r", &Simulation::SelfCollisionInformation::r)
          .def_readonly("f1", &Simulation::SelfCollisionInformation::f1)
          .def_readonly("f2", &Simulation::SelfCollisionInformation::f2)
          .def_readonly("m1", &Simulation::SelfCollisionInformation::m1)
          .def_readonly("m2", &Simulation::SelfCollisionInformation::m2)
          .def_readonly("normal", &Simulation::SelfCollisionInformation::normal)
          .def_readonly("v2f", &Simulation::SelfCollisionInformation::v2f)
          .def_readonly("e2e", &Simulation::SelfCollisionInformation::e2e)
          .def_readonly("dist", &Simulation::SelfCollisionInformation::dist)
          .def_readonly("layerId", &Simulation::SelfCollisionInformation::layerId)
          .def_readonly("type", &Simulation::SelfCollisionInformation::type);

  // Simulation::ForwardInformation
  py::class_<Simulation::ForwardInformation>(m, "ForwardInformation")
          .def_readonly("x", &Simulation::ForwardInformation::x)
          .def_readonly("stepIdx", &Simulation::ForwardInformation::stepIdx)
          .def_readonly("sysMatId", &Simulation::ForwardInformation::sysMatId)
          .def_readonly("M", &Simulation::ForwardInformation::M)
          .def_readonly("t", &Simulation::ForwardInformation::t)
          .def_readonly("v", &Simulation::ForwardInformation::v)
          .def_readonly("x_prev", &Simulation::ForwardInformation::x_prev)
          .def_readonly("v_prev", &Simulation::ForwardInformation::v_prev)
          .def_readonly("f", &Simulation::ForwardInformation::f)
          .def_readonly("r", &Simulation::ForwardInformation::r)
          .def_readonly("x_fixedpoints", &Simulation::ForwardInformation::x_fixedpoints)
          .def_readonly("avgDeformation", &Simulation::ForwardInformation::avgDeformation)
          .def_readonly("maxDeformation", &Simulation::ForwardInformation::maxDeformation)
          .def_readonly("collisionInfos", &Simulation::ForwardInformation::collisionInfos);


  // Simulation::BackwardInformation
  py::class_<Simulation::BackwardInformation>(m, "BackwardInformation")
          .def_readonly("dL_dx", &Simulation::BackwardInformation::dL_dx)
          .def_readonly("dL_dv", &Simulation::BackwardInformation::dL_dv)
          .def_readonly("dL_dfext", &Simulation::BackwardInformation::dL_dfext)
          .def_readonly("dL_dxfixed", &Simulation::BackwardInformation::dL_dxfixed)
          .def_readonly("dL_dwind", &Simulation::BackwardInformation::dL_dwind)
          .def_readonly("dL_ddensity", &Simulation::BackwardInformation::dL_ddensity)
          .def_readonly("dL_dk_pertype", &Simulation::BackwardInformation::dL_dk_pertype)
          .def_readonly("dL_dsplines", &Simulation::BackwardInformation::dL_dsplines)
          .def_readonly("dL_dmu", &Simulation::BackwardInformation::dL_dmu)
          .def_readonly("loss", &Simulation::BackwardInformation::loss)
          .def_readonly("totalRuntime", &Simulation::BackwardInformation::totalRuntime)
          .def_readonly("converged", &Simulation::BackwardInformation::converged)
          .def_readonly("convergedAccum", &Simulation::BackwardInformation::convergedAccum)
          .def_readonly("backwardIters", &Simulation::BackwardInformation::backwardIters)
          .def_readonly("backwardTotalIters", &Simulation::BackwardInformation::backwardTotalIters);


  // Simulation::BackwardTaskInformation
  py::class_<Simulation::BackwardTaskInformation>(m, "BackwardTaskInformation")
          .def_readonly("dL_dk_pertype", &Simulation::BackwardTaskInformation::dL_dk_pertype)
          .def_readonly("dL_density", &Simulation::BackwardTaskInformation::dL_density)
          .def_readonly("dL_dfext", &Simulation::BackwardTaskInformation::dL_dfext)
          .def_readonly("dL_dfwind", &Simulation::BackwardTaskInformation::dL_dfwind)
          .def_readonly("adddr_dd", &Simulation::BackwardTaskInformation::adddr_dd)
          .def_readonly("dL_dcontrolPoints", &Simulation::BackwardTaskInformation::dL_dcontrolPoints)
          .def_readonly("dL_dmu", &Simulation::BackwardTaskInformation::dL_dmu)
          .def_readonly("dL_dx0", &Simulation::BackwardTaskInformation::dL_dx0)
          .def_readonly("dL_dwindFactor", &Simulation::BackwardTaskInformation::dL_dwindFactor)
          .def_readonly("forwardAccuracyLevel", &Simulation::BackwardTaskInformation::forwardAccuracyLevel)
          .def_readonly("backwardAccuracyLevel", &Simulation::BackwardTaskInformation::backwardAccuracyLevel)
          .def_readonly("randSeed", &Simulation::BackwardTaskInformation::randSeed)
          .def_readonly("srandSeed", &Simulation::BackwardTaskInformation::srandSeed);

  // Simulation::CorresPondenceTargetInfo
  py::class_<Simulation::CorresPondenceTargetInfo>(m, "CorresPondenceTargetInfo")
          .def_readonly("frameIdx", &Simulation::CorresPondenceTargetInfo::frameIdx)
          .def_readonly("targetPos", &Simulation::CorresPondenceTargetInfo::targetPos)
          .def_readonly("particleIndices", &Simulation::CorresPondenceTargetInfo::particleIndices);

  // Simulation::LossInfo
  py::class_<Simulation::LossInfo>(m, "LossInfo")
          .def_readwrite("targetLoc", &Simulation::LossInfo::targetLoc)
          .def_readwrite("targetTranslation", &Simulation::LossInfo::targetTranslation)
          .def_readwrite("targetFrameShape", &Simulation::LossInfo::targetFrameShape)
          .def_readwrite("targetPosPairs", &Simulation::LossInfo::targetPosPairs);
  // .def_readonly("targetSimulation", &Simulation::targetSimulation)
  // .def_readonly("target")

  py::class_<Simulation::ParamInfo>(m, "ParamInfo")
          .def_readwrite("x0", &Simulation::ParamInfo::x0)
          .def_readwrite("v0", &Simulation::ParamInfo::v0)
          .def_readwrite("f_ext", &Simulation::ParamInfo::f_ext)
          .def_readwrite("f_extwind", &Simulation::ParamInfo::f_extwind)
          .def_readwrite("density", &Simulation::ParamInfo::density)
          .def_readonly("k_pertype", &Simulation::ParamInfo::k_pertype);

  // Primitive
  py::class_<Primitive> primitive(m, "Primitive");
  py::enum_<Primitive::PrimitiveType>(primitive, "PrimitiveType")
          .value("PLANE", Primitive::PrimitiveType::PLANE)
          .value("CUBE", Primitive::PrimitiveType::CUBE)
          .value("SPHERE", Primitive::PrimitiveType::SPHERE)
          .value("CAPSULE", Primitive::PrimitiveType::CAPSULE)
          .value("FOOT", Primitive::PrimitiveType::FOOT)
          .value("LOWER_LEG", Primitive::PrimitiveType::LOWER_LEG)
          .value("BOWL", Primitive::PrimitiveType::BOWL)
          .value("HANGER", Primitive::PrimitiveType::HANGER)
          .value("HANGER2", Primitive::PrimitiveType::HANGER2)
          .export_values();

  primitive.def_readwrite("primitives", &Primitive::primitives)
          .def_readwrite("isPrimitiveCollection", &Primitive::isPrimitiveCollection)
          .def_readwrite("isInCollision", &Primitive::isInCollision)
          .def_readwrite("mesh", &Primitive::mesh)
          .def_readwrite("points", &Primitive::points)
          .def_readwrite("type", &Primitive::type)
          .def_readwrite("center", &Primitive::center)
          .def_readwrite("centerInit", &Primitive::centerInit)
          .def_readwrite("velocity", &Primitive::velocity)
          .def_readwrite("velocityInit", &Primitive::velocityInit)
          .def_readwrite("angular", &Primitive::angular)
          .def_readwrite("angularInit", &Primitive::angularInit)
          .def_readwrite("angularVelocity", &Primitive::angularVelocity)
          .def_readwrite("angularVelocityInit", &Primitive::angularVelocityInit)
          .def_readwrite("rotation_matrix", &Primitive::rotation_matrix)
          .def_readwrite("rotation_matrixInit", &Primitive::rotation_matrixInit)
          .def_readwrite("mu", &Primitive::mu)

          .def_readonly_static("primitiveTypeStrings", &Primitive::primitiveTypeStrings)
          .def("getMesh", &Primitive::getMesh, "getMesh", py::arg("cumulativeMesh"), py::arg("cumulativePos"), py::arg("center"), py::arg("frameIdx") )
          .def("getPointVec", &Primitive::getPointVec, "getPointVec")
          .def_readwrite("forwardRecords", &Primitive::forwardRecords)
          .def_readwrite("rotationRecords", &Primitive::rotationRecords);

  py::class_<Particle>(m, "Particle")
          .def_readwrite("pos", &Particle::pos)
          .def_readwrite("pos_init", &Particle::pos_init)
          .def_readwrite("idx", &Particle::idx)
          .def_readwrite("mass", &Particle::mass);

  // Simulation
  py::class_<Simulation>(m, "Simulation")
          .def_readonly("taskLossInfo", &Simulation::taskLossInfo)
          .def_readonly("primitives", &Simulation::primitives)
          .def_readonly("sceneConfig", &Simulation::sceneConfig)
          .def_readwrite("forwardRecords", &Simulation::forwardRecords)
          .def_readwrite("useCustomRLFixedPoint", &Simulation::useCustomRLFixedPoint)
          .def_readwrite("particles", &Simulation::particles)
          .def_readwrite("perStepGradient", &Simulation::perStepGradient)
          .def_readwrite("gradientClipping", &Simulation::gradientClipping)
          .def_readwrite("gradientClippingThreshold", &Simulation::gradientClippingThreshold)
          .def_property_readonly("ndof_u", &Simulation::getActionDim)
          .def_property_readonly("num_particles", &Simulation::getNumParticles)
           .def_readwrite_static("forwardConvergenceThreshold", &Simulation::forwardConvergenceThreshold)
           .def_readwrite_static("backwardConvergenceThreshold", &Simulation::backwardConvergenceThreshold)
          .def("resetSystem", static_cast<void (Simulation::*)()>(&Simulation::resetSystem),
               "reset the simulation")
          .def("step", &Simulation::step,
               "forward one step")
          .def("getCurrentPosVelocityVec", &Simulation::getCurrentPosVelocityVec, "get posvel vecs")
          .def("appendPerStepGradient", &Simulation::appendPerStepGradient, "append grad", py::arg("x"))
          .def("stepNN", &Simulation::stepNN, "forward one step with arg", py::arg("idx"), py::arg("x"), py::arg("v"), py::arg("fixedPointPos"))
          .def("stepCouple", &Simulation::stepCouple, "forward one step with arg", py::arg("idx"), py::arg("x"), py::arg("v"), py::arg("fixedPointPos"), py::arg("coupleForce"))
          .def("setWindAndCollision", &Simulation::setWindAncCollision, "setWindAndCollision", py::arg("windEnable"), py::arg("collisionEnable"), py::arg("selfCollisionEnable"), py::arg("enableConstantForcefield"))
          .def("getStateInfo", &Simulation::getStateInfo,
               "get the forward info of the current step")
          .def("setAction", &Simulation::setAction,
               "set the target position for clips")
          .def("exportCurrentMeshPos", &Simulation::exportCurrentMeshPos,
               "export the mesh at certain step",
               py::arg("step"), py::arg("filename"))
          .def("setPrintVerbose", &Simulation::setPrintVerbose,
               "set whether to print verbose info",
               py::arg("flag"))
          .def("getPastStateInfo", &Simulation::getPastStateInfo,
               "get the forward info of the current step of a past time step",
               py::arg("stepIdx"))
          .def("exportCurrentSimulation", &Simulation::exportCurrentSimulation,
               "export the simulation to files",
               py::arg("fileName"))
          .def("stepBackward", &Simulation::stepBackward, "stepbackward one step", py::arg("taskInfo"), py::arg("dL_dxvfnew"), py::arg("forwardInfo_new"), py::arg("isStart"), py::arg("dL_dxinit"), py::arg("dL_dvinit"))
          .def("stepBackwardNN", &Simulation::stepBackwardNN, "stepbackward one step", py::arg("taskInfo"), py::arg("dL_dxnew"),  py::arg("dL_dvnew"), py::arg("forwardInfo_new"), py::arg("isStart"), py::arg("dL_dxinit"), py::arg("dL_dvinit"))
          ;

  // Optimization helper
  py::class_<OptimizeHelper>(m, "OptimizeHelper")
          .def_readonly("paramLowerBound", &OptimizeHelper::paramLowerBound)
          .def_readonly("paramUpperBound", &OptimizeHelper::paramUpperBound)
          .def_readonly("forward_steps", &OptimizeHelper::FORWARD_STEPS)
          .def_readonly("sim", &OptimizeHelper::system)
          .def_readonly("paramLogScaleTransformOn", &OptimizeHelper::paramLogScaleTransformOn)
          .def_readonly("taskInfo", &OptimizeHelper::taskInfo)
          .def_readonly("lossType", &OptimizeHelper::lossType)
          .def_readonly("lossInfo", &OptimizeHelper::lossInfo)
          .def_readonly("paramActual", &OptimizeHelper::param_actual)
          .def("getActualParam", &OptimizeHelper::getActualParam,
               "getactualparam")
          .def("getRandomParam", &OptimizeHelper::getRandomParam,
               "generate random initial parameters",
               py::arg("randSeed") = 0)
          .def("runSimulationAndGetLoss", &OptimizeHelper::runSimulationAndGetLoss,
               "compute loss from parameter vector",
               py::arg("x"))
          .def("vecXdToParamInfo", &OptimizeHelper::vecXdToParamInfo,
               "compute loss from parameter vector",
               py::arg("x"))
               .def("gradientInfoToVecXd", &OptimizeHelper::gradientInfoToVecXd,
                    "convert grad struct to grad vector",
                    py::arg("grad"))
          .def("runSimulationAndGetLossGradient", &OptimizeHelper::runSimulationAndGetLossAndGradients,
               "compute loss and grads from parameter vector",
               py::arg("x"))
               ;

  m.def("makeCustomizedSim", &makeCustomizedSim, "initialize a simulation instance", py::arg("exampleName"), py::arg("runBackward") = true, py::arg("config"));
  m.def("makeSim", &makeSim, "initialize a simulation instance", py::arg("exampleName"), py::arg("runBackward") = true);

  m.def("makeOptimizeHelper", &makeOptimizeHelper,
        "initialize an optimize helper", py::arg("exampleName") );

  m.def("makeOptimizeHelperWithSim", &makeOptimizeHelperWithSim,
        "initialize an optimize helper", py::arg("exampleName"),  py::arg("sim") );

  m.def("enableOpenMP", &enableOpenMP, "set up Open MP", py::arg("n_threads") = 5);

  m.def("render", &render, "rendering the previous trajectry", py::arg("sim"), py::arg("renderPosPairs") = false, py::arg("autoExit") = true);
}