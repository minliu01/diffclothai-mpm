//
// Created by Yifei Li on 10/29/20.
// Email: liyifei@csail.mit.edu
//

#ifndef OMEGAENGINE_PRIMITIVE_H
#define OMEGAENGINE_PRIMITIVE_H

#include "Triangle.h"
#include <Eigen/src/Geometry/Transform.h>
#include "../engine/UtilityFunctions.h"
#include "../engine/Constants.h"
//#include <fcl/fcl.h>
#include "../../../external/fcl/build/include/fcl/fcl.h"

class Primitive {
public:
    enum PrimitiveType {
        PLANE, CUBE, SPHERE, CAPSULE, PINBALL, FOOT, LOWER_LEG, BOWL, DISK, TABLE, HANGER, HANGER2, GRIPPER
    };

    std::vector<Primitive *> primitives;
    static std::vector<std::string> primitiveTypeStrings;
    static Vec3d gravity;
    Vec3d center, centerInit, velocity, velocityInit; // center is relative to parent primitive, if there is a parent
    Vec3d angular, angularInit, angularVelocity, angularVelocityInit;
    Mat3x3d rotation_matrix, rotation_matrixInit;
    double mass;
    bool isEnabled;
    bool simulationEnabled;
    bool isPrimitiveCollection;
    bool isStaitc;
    bool isInCollision;
    bool gravityEnabled;
    double mu;
    Vec3d color;
    std::vector<Triangle> mesh;
    std::vector<Particle> points;
    PrimitiveType type;
    std::vector<Vec3d> forwardRecords;
    std::vector<Mat3x3d> rotationRecords;


    Primitive(PrimitiveType t, const Vec3d &c, bool isCollection, Vec3d color) : mu(10.0), type(t), center(c),
                                                                                 centerInit(c),
                                                                                 mass(30),
                                                                                 velocity(0, 0, 0),
                                                                                 velocityInit(0, 0, 0),
                                                                                 angular(0, 0, 0),
                                                                                 angularInit(0, 0, 0),
                                                                                 angularVelocity(0, 0, 0),
                                                                                 angularVelocityInit(0, 0, 0),
                                                                                 rotation_matrix(Mat3x3d::Identity()),
                                                                                 rotation_matrixInit(Mat3x3d::Identity()),
                                                                                 isEnabled(false),
                                                                                 gravityEnabled(false),
                                                                                 isStaitc(true),
                                                                                 isInCollision(true),
                                                                                 simulationEnabled(true),
                                                                                 isPrimitiveCollection(isCollection),
                                                                                 color(color) {}

    Primitive(Primitive &&other) {
      center = other.center;
      centerInit = other.centerInit;
      mesh = std::move(other.mesh);
      points = std::move(other.points);
      type = other.type;
      isPrimitiveCollection = other.isPrimitiveCollection;
      mass = other.mass;
    }

    void updateForwardRecord() {
      forwardRecords.push_back(center);
      rotationRecords.push_back(rotation_matrix);
      if (isPrimitiveCollection) {
        for (Primitive* p : primitives)
          p->updateForwardRecord();
      }
    }

    void getMesh(std::vector<Triangle>& cumulativeMesh, VecXd& cumulativePos, Vec3d center, int frameIdx) {
      Vec3d centerPos = forwardRecords[frameIdx];
      if (isPrimitiveCollection) {
        for (Primitive* p : primitives) {
          p->getMesh(cumulativeMesh, cumulativePos,center + centerPos, frameIdx);
        }
      } else {
        updateToPose(cumulativeMesh, cumulativePos, mesh, points, center + centerPos);
      }
    }

    VecXd getPointVec() {
      VecXd out(points.size() * 3);
      out.setZero();
      for (int i = 0; i < points.size(); i++) {
        out.segment( i * 3, 3) = points[i].pos + center;
      }
      return out;
    }

    void updateToPose(std::vector<Triangle>& cumulativeMesh, VecXd& cumulativePos, std::vector<Triangle>& mesh,  std::vector<Particle>& points,  Vec3d center) {
      int prevParticleNum = cumulativePos.rows() / 3;
      int newParticleNum = points.size() + prevParticleNum;

      VecXd newCumulativePos(newParticleNum * 3);
      newCumulativePos.setZero();
      newCumulativePos.segment(0,  cumulativePos.rows()) = cumulativePos;

      int posOffset =  cumulativePos.rows();
      for (int i = 0; i < points.size(); i++) {
        newCumulativePos.segment(posOffset + i * 3, 3) = points[i].pos + center;
      }

      for (int i = 0; i < mesh.size(); i++) {
        Triangle t = mesh[i];
        t.p0_idx += prevParticleNum;
        t.p1_idx += prevParticleNum;
        t.p2_idx += prevParticleNum;
        cumulativeMesh.emplace_back(t);
      }

      cumulativePos = newCumulativePos;
    }

    virtual Vec3d getForces(double timeStep) {
      if (gravityEnabled)
        return gravity * mass;
      return Vec3d(0,0,0);

    }

    virtual void step(double timeStep) {
      if (!isStaitc) {
        if (gravityEnabled)
          velocity += gravity * timeStep;
        center += timeStep * velocity;
      }

    }

    virtual ~Primitive() {}


    virtual bool isInContact(const Vec3d &center_prim, const Vec3d &pos, const Vec3d &velocity, Vec3d &normal,
                             double &dist, Vec3d &v_out) { return false; }
    virtual bool isInContact(const Vec3d &center_prim, const Vec3d &pos, const Vec3d &velocity, Vec3d &normal,
                             double &dist, Vec3d &v_out, Vec3d &contact_point) { return false; }

    static Vec3d getSpherePos(double radius, double phi, double theta) {
      double x = radius * glm::cos(glm::radians(phi)) * glm::sin(glm::radians(theta));
      double y = radius * glm::sin(glm::radians(phi)) * glm::sin(glm::radians(theta));
      double z = radius * glm::cos(glm::radians(theta));
      return Vec3d(x, y, z);
    }


    void setEnabled(bool v) {
      isEnabled = v;
    }

    void setSimulationEnabled(bool v) {
      simulationEnabled = v;
    }

    void setMu(double v) {
      mu = v;
    }

    void reset() {
      center = centerInit;
      velocity = velocityInit;
      angular = angularInit;
      angularVelocity = angularVelocityInit;
      rotation_matrix = rotation_matrixInit;
      forwardRecords.clear();
      forwardRecords.push_back(center);
      rotationRecords.clear();
      rotationRecords.push_back(rotation_matrix);
      if (isPrimitiveCollection) {
        for (Primitive* p : primitives) {
          p->reset();
        }
      }
    }

    static std::pair<bool, Vec3d> pointInsideTriangle( Triangle t, Vec3d p_prime) {
      Vec3d AB = (t.p1()->pos - t.p0()->pos);
      Vec3d AC = (t.p2()->pos - t.p0()->pos);
      Vec3d n = AB.cross(AC);
      double n2 = n.squaredNorm() ;
      Vec3d AP = p_prime - t.p0()->pos;
      double alpha = AB.cross(AP).dot(n) / n2;
      double beta = AP.cross(AC).dot(n) / n2;
      double gamma = 1 - alpha - beta;

      bool isInside = (alpha >= 0) && (beta >= 0) && (gamma >= 0) && (gamma <= 1) && (alpha <= 1) && (beta <= 1);

      Vec3d proj = alpha * t.p1()->pos + beta * t.p2()->pos + gamma * t.p0()->pos;
      return std::make_pair(isInside, proj);
    }

    static double distanceToPlane(Vec3d &normal, double d, Vec3d &p) {
      double dist = (normal.dot(p) + d) / normal.norm();
      return dist;
    }


    static std::pair<Vec3d, double> projectionOnLine(Vec3d a, Vec3d b, Vec3d p) {
      Vec3d AP = p - a;
      Vec3d AB = b - a;
      Vec3d proj = AP.dot(AB) / AB.dot(AB) * AB;
      Vec3d P_prime = a + proj;
      double AB_l = (b - a).norm();
      double AP_l = (P_prime - a).norm();
      double PB_l = (P_prime - b).norm();
      double t = AP_l / AB_l;
      if (PB_l > AB_l)
        t *= -1;

      return std::make_pair(P_prime, t);
    }

    static Vec3d projectionOnPlane(Vec3d normal, double d, Vec3d p) {
      return (p - (normal.dot(p) + d) * normal);
    }
};


class Sphere : public Primitive {
public:
    double radius;
    bool rotates, discretized;
    Mat3x3d rotMax = (Eigen::AngleAxis<double>(0, Vec3d::UnitX())
                      * Eigen::AngleAxis<double>(0.5 * 3.14159, Vec3d::UnitY())
                      * Eigen::AngleAxis<double>(0, Vec3d::UnitZ())).toRotationMatrix();

    Sphere(Sphere &&other) : Primitive(std::move(other)) {
      radius = other.radius;
    }

    Sphere &operator=(Sphere &&other) {
      radius = other.radius;
      return *this;
    }


    Sphere(Vec3d center, double radius, Vec3d myColor = Vec3d(1, 0, 0), int resolution = 40, bool discretized = false);

    void step(double timeStep) override {
      center += timeStep * velocity;
    }

    bool isInContact(const Vec3d &center_prim, const Vec3d &pos, const Vec3d &velocity, Vec3d &normal,
                     double &dist, Vec3d &v_out) override;
    bool isInContact(const Vec3d &center_prim, const Vec3d &pos, const Vec3d &velocity, Vec3d &normal,
                     double &dist, Vec3d &v_out, Vec3d &contact_point) override;
};


class Bowl : public Primitive {
public:
    double radius;

    Bowl(Bowl &&other) : Primitive(std::move(other)) {
      radius = other.radius;
    }

    Bowl &operator=(Bowl &&other) {
      radius = other.radius;
      return *this;
    }

    Bowl(Vec3d center, double radius, Vec3d myColor = Vec3d(1, 0, 0), int resolution = 40);

    bool isInContact(const Vec3d &center_prim, const Vec3d &pos, const Vec3d &velocity, Vec3d &normal,
                     double &dist, Vec3d &v_out) override;
};


class Capsule : public Primitive {
public:
    double radius;
    double length;
    Vec3d axis, parentAxis, globalAxis; // axis is relative to parentAxis,  globalAxis is the actual direciton of the joint
    Eigen::Transform<double, 3, Eigen::Affine> rotationFromParent, globalRotation;
    Capsule* parent;
    Primitive* rootPrimitiveContainer;
    bool isRoot;
    Capsule(Vec3d anchorCenter, double radius, double length, Vec3d parentAxis, Vec3d axis, Capsule* parent = nullptr, bool isRoot = true,
            Primitive* rootPrimitiveContainer = nullptr,
            Vec3d myColor = Vec3d(0.671993, 0.395403, 0.0829298), int resSphere = 40,
            int resBody = 40);

    Vec3d getTransformedPosFromJointBindPos(Vec3d original) {
        Rotation accumulated;
        Vec3d accumCenter;
        accumCenter.setZero();
        accumulated.setIdentity();
        Capsule* joint = this;
        while (!joint->isRoot) {
            accumCenter += joint->center;
            accumulated = accumulated * joint->rotationFromParent;
            joint = joint->parent;
        };

        accumCenter += joint->center;
        accumulated = accumulated * joint->rotationFromParent;
        if (joint->rootPrimitiveContainer != nullptr)
            accumCenter += joint->rootPrimitiveContainer->center;

        return accumCenter +  globalRotation * original;
    }

    bool isInContact(const Vec3d &center_prim, const Vec3d &pos, const Vec3d &velocity, Vec3d &normal,
                     double &dist, Vec3d &v_out) override;

    void step(double timeStep) override {
      center += timeStep * velocity;
    }
};


class Foot : public Primitive {
public:
//    Capsule* toes[5];
    Capsule* footBase;

    Foot(Vec3d center, Rotation accumRot, Vec3d axis, double toeLength, double footLength, Vec3d color);
    ~Foot() {
     delete footBase;
   }

    bool isInContact(const Vec3d &center_prim, const Vec3d &pos, const Vec3d &velocity, Vec3d &normal,
                     double &dist, Vec3d &v_out) override;

    void step(double timeStep) override {
      center += timeStep * velocity;
    }
};


class LowerLeg : public Primitive {
public:
    Capsule  *leg, *foot;
    Sphere* joint;
    Vec3d axis;
    Rotation rotation;
//    Vec3d footUp, legUp;
    LowerLeg(Vec3d center, Vec3d axis, double legLength, double footLength);

    bool isInContact(const Vec3d &center_prim, const Vec3d &pos, const Vec3d &velocity, Vec3d &normal,
                     double &dist, Vec3d &v_out) override;

    void step(double timeStep) override {
      center += timeStep * velocity;
    }

    void createNewMesh(Vec3d center, Vec3d axis, double legLength, double footLength) {
      if (leg) {
        delete leg;
        delete foot;
        delete joint;
      }

      primitives.clear();
      primitives.reserve(2);
      rotation.setIdentity();
      axis.normalize();
      double radius = 0.8;
      double toeLength = 0.8;

      foot = new Capsule(Vec3d(0, 0, 0), radius, footLength, Vec3d(0, 1, 0), axis, nullptr, true, this, COLOR_WOOD);
      Vec3d legCenter = foot->rotationFromParent * Vec3d(0, foot->length, 0);

      leg = new Capsule(legCenter, radius, legLength, axis, Vec3d(0, 0.7, 0.3), foot, false, nullptr, COLOR_WOOD);
      joint = new Sphere(foot->rotationFromParent * Vec3d(0, foot->length, 0), radius + 0.05, COLOR_WOOD);

      primitives.emplace_back(joint);
      primitives.emplace_back(foot); // leftLeg l=5
      primitives.emplace_back(leg); // foot l=4
    }

    ~LowerLeg() {
      delete leg;
      delete foot;
      delete joint;
    }
};


class Plane : public Primitive {
public:
    double width, height, boundaryRadius, thickness;
    Vec3d planeNormal;
    double d, planeNormalNorm;
    Vec3d lowerLeft, lowerRight;
    Vec3d upperLeft, upperRight;

    Plane(Vec3d center, Vec3d upperLeft, Vec3d upperRight, Vec3d color = Vec3d(0, 0, 1));

    double distanceToPlane(Vec3d p) {
      double dist =  (planeNormal.dot(p) + d) / planeNormalNorm;
      return dist;
    }


    bool isInContact(const Vec3d &center_prim, const Vec3d &pos, const Vec3d &velocity, Vec3d &normal,
                     double &dist, Vec3d &v_out) override;
    bool isInContact(const Vec3d &center_prim, const Vec3d &pos, const Vec3d &velocity, Vec3d &normal,
                     double &dist, Vec3d &v_out, Vec3d &contact_point) override;

    bool sphereIsInContact(const Vec3d &center_prim, const Vec3d &sphereCenter, double sphereRadius);
    void step(double timeStep) override {
      center += timeStep * velocity;
    }
};


class Hanger : public Primitive {
public:
    double scale;
    Plane *left_plane, *right_plane;
    Plane *top_plane, *bottom_plane;
    Plane *front_plane, *back_plane;
    Plane *left2_plane, *right2_plane;
    Plane *top2_plane;
    Plane *front2_plane, *back2_plane;
    Plane *underground;

    Hanger(Vec3d center, double w, double l, double h, double l2, double h2);

    bool isInContact(const Vec3d &center_prim, const Vec3d &pos, const Vec3d &velocity, Vec3d &normal,
                     double &dist, Vec3d &v_out) override;
    
    void createNewMesh(Vec3d center, double w, double l, double h, double l2, double h2) {
      if (left_plane) {
        delete left_plane;
        delete right_plane;
        delete top_plane;
        delete bottom_plane;
        delete front_plane;
        delete back_plane;
        delete left2_plane;
        delete right2_plane;
        delete top2_plane;
        delete front2_plane;
        delete back2_plane;
        delete underground;
      }

      primitives.clear();
      primitives.reserve(12);

      Vec3d center2 = center + Vec3d(0.,h+h2,0.);

      left_plane = new Plane(center+Vec3d(0.,0.,-w), center+Vec3d(l,h,-w), center+Vec3d(-l,h,-w), COLOR_WOOD);
      right_plane = new Plane(center+Vec3d(0.,0.,w), center+Vec3d(-l,h,w), center+Vec3d(l,h,w), COLOR_WOOD);

      top_plane = new Plane(center+Vec3d(0.,h,0.), center+Vec3d(l,h,-w), center+Vec3d(l,h,w), COLOR_WOOD);
      bottom_plane = new Plane(center+Vec3d(0.,-h,0.), center+Vec3d(l,-h,w), center+Vec3d(l,-h,-w), COLOR_WOOD);

      front_plane = new Plane(center+Vec3d(-l,0.,0.), center+Vec3d(-l,h,-w), center+Vec3d(-l,h,w), COLOR_WOOD);
      back_plane = new Plane(center+Vec3d(l,0.,0.), center+Vec3d(l,h,w), center+Vec3d(l,h,-w), COLOR_WOOD);

      left2_plane = new Plane(center2+Vec3d(0.,0.,-w), center2+Vec3d(l2,h2,-w), center2+Vec3d(-l2,h2,-w), COLOR_WOOD);
      right2_plane = new Plane(center2+Vec3d(0.,0.,w), center2+Vec3d(-l2,h2,w), center2+Vec3d(l2,h2,w), COLOR_WOOD);

      top2_plane = new Plane(center2+Vec3d(0.,h2,0.), center2+Vec3d(l2,h2,-w), center2+Vec3d(l2,h2,w), COLOR_WOOD);

      front2_plane = new Plane(center2+Vec3d(-l2,0.,0.), center2+Vec3d(-l2,h2,-w), center2+Vec3d(-l2,h2,w), COLOR_WOOD);
      back2_plane = new Plane(center2+Vec3d(l2,0.,0.), center2+Vec3d(l2,h2,w), center2+Vec3d(l2,h2,-w), COLOR_WOOD);

      underground = new Plane(Vec3d(0,-1.5,0), Vec3d(7,-1.5,7), Vec3d(-7,-1.5,7), COLOR_GRAY50);

      primitives.emplace_back(underground);
      primitives.emplace_back(back2_plane);
      primitives.emplace_back(front2_plane);
      primitives.emplace_back(top2_plane);
      primitives.emplace_back(right2_plane);
      primitives.emplace_back(left2_plane);
      primitives.emplace_back(back_plane);
      primitives.emplace_back(front_plane);
      primitives.emplace_back(bottom_plane);
      primitives.emplace_back(top_plane);
      primitives.emplace_back(right_plane);
      primitives.emplace_back(left_plane);
    }

    void step(double timeStep) override {
      center += timeStep * velocity;
    }

    ~Hanger(){
      delete left_plane;
      delete right_plane;
      delete top_plane;
      delete bottom_plane;
      delete front_plane;
      delete back_plane;
      delete left2_plane;
      delete right2_plane;
      delete top2_plane;
      delete front2_plane;
      delete back2_plane;
      delete underground;
    }
};


class Hanger2 : public Primitive {
public:
    std::vector<fcl::Vector3d> vertices;
    std::vector<fcl::Triangle> triangles;
    typedef fcl::BVHModel<fcl::OBBRSSd> Model;
    std::shared_ptr<Model> model;
    Hanger2(Vec3d center, double scale, const char *filename, Vec3d color = COLOR_GRAY57);

    bool isInContact(const Vec3d &center_prim, const Vec3d &pos, const Vec3d &velocity, Vec3d &normal,
                     double &dist, Vec3d &v_out, Vec3d &contact_point) override;

    void step(double timeStep) override {
      // do nothing here !!!
      // center += timeStep * velocity;
      // angular += timeStep * angularVelocity;
    }
};

class Cube : public Primitive {
public:
    Plane *left_plane, *right_plane;
    Plane *top_plane, *bottom_plane;
    Plane *front_plane, *back_plane;

    Cube(Vec3d center, double l, double h, double w, Vec3d color = COLOR_GRAY57);

    bool isInContact(const Vec3d &center_prim, const Vec3d &pos, const Vec3d &velocity, Vec3d &normal,
                     double &dist, Vec3d &v_out) override;
    bool isInContact(const Vec3d &center_prim, const Vec3d &pos, const Vec3d &velocity, Vec3d &normal,
                     double &dist, Vec3d &v_out, Vec3d &contact_point) override;
    
    void createNewMesh(Vec3d center, double l, double h, double w) {
      if (left_plane) {
        delete left_plane;
        delete right_plane;
        delete top_plane;
        delete bottom_plane;
        delete front_plane;
        delete back_plane;
      }

      primitives.clear();
      primitives.reserve(6);

      left_plane = new Plane(center+Vec3d(0.,0.,-w), center+Vec3d(l,h,-w), center+Vec3d(-l,h,-w), COLOR_GRAY50);
      right_plane = new Plane(center+Vec3d(0.,0.,w), center+Vec3d(-l,h,w), center+Vec3d(l,h,w), COLOR_GRAY50);

      top_plane = new Plane(center+Vec3d(0.,h,0.), center+Vec3d(l,h,-w), center+Vec3d(l,h,w), COLOR_GRAY50);
      bottom_plane = new Plane(center+Vec3d(0.,-h,0.), center+Vec3d(l,-h,w), center+Vec3d(l,-h,-w), COLOR_GRAY50);

      front_plane = new Plane(center+Vec3d(-l,0.,0.), center+Vec3d(-l,h,-w), center+Vec3d(-l,h,w), COLOR_GRAY50);
      back_plane = new Plane(center+Vec3d(l,0.,0.), center+Vec3d(l,h,w), center+Vec3d(l,h,-w), COLOR_GRAY50);

      primitives.emplace_back(back_plane);
      primitives.emplace_back(front_plane);
      primitives.emplace_back(bottom_plane);
      primitives.emplace_back(top_plane);
      primitives.emplace_back(right_plane);
      primitives.emplace_back(left_plane);
    }

    void step(double timeStep) override {
      center += timeStep * velocity;
    }

    ~Cube(){
      delete left_plane;
      delete right_plane;
      delete top_plane;
      delete bottom_plane;
      delete front_plane;
      delete back_plane;
    }
};

#endif //OMEGAENGINE_PRIMITIVE_H
