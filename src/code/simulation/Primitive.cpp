//
// Created by Yifei Li on 10/29/20.
// Email: liyifei@csail.mit.edu
//

#include "Primitive.h"
#include "../engine/Constants.h"
#include "../engine/MeshFileHandler.h"
#include <Eigen/src/Geometry/Transform.h>
//#include <fcl/fcl.h>
//#include "../../../../fcl/build/include/fcl/fcl.h"
#include "../../../external/fcl/test/test_fcl_utility.h"

int num_max_contacts = std::numeric_limits<int>::max();

std::vector<std::string> Primitive::primitiveTypeStrings =
    std::vector<std::string>{
        "PLANE", "CUBE", "SPHERE", "CAPSULE", "PINBALL", "FOOT",   "LOWER_LEG",
        "BOWL",  "DISK", "TABLE",  "HANGER",  "HANGER2", "GRIPPER"};
Vec3d Primitive::gravity(0, -9.8, 0);

int N2 = 1;
double ret2[100];

double *parse_config2() {
  std::ifstream infile("/home/ubuntu/diffcloth/src/code/simulation/config.txt");
  for (int i = 0; i < N2; i++) {
    char data[100];
    infile >> data;
    ret2[i] = std::stod(data);
  }
  return ret2;
}

Plane::Plane(Vec3d center, Vec3d upperLeft, Vec3d upperRight, Vec3d color)
    : Primitive(PLANE, center, false, color), thickness(5) {
  width = (upperLeft - upperRight).norm();
  upperLeft -= center;
  upperRight -= center;
  this->upperLeft = upperLeft;
  this->upperRight = upperRight;
  lowerRight = -upperLeft;
  lowerLeft = -upperRight;
  boundaryRadius = std::max(upperLeft.norm(), upperRight.norm());
  points.emplace_back(1, upperLeft, upperLeft, Vec3d(0, 0, 0), Vec2i(0, 0), 0);
  points.emplace_back(1, upperRight, upperRight, Vec3d(0, 0, 0), Vec2i(0, 1),
                      1);
  points.emplace_back(1, lowerLeft, lowerLeft, Vec3d(0, 0, 0), Vec2i(1, 0), 2);
  points.emplace_back(1, lowerRight, lowerRight, Vec3d(0, 0, 0), Vec2i(1, 1),
                      3);

  mesh.emplace_back(0, 1, 2, points);
  mesh[0].overrideColor = true;
  mesh[0].color = color;

  mesh.emplace_back(2, 1, 3, points);
  mesh[1].overrideColor = true;
  mesh[1].color = color;

  planeNormal = (upperRight).cross(upperLeft).normalized();
  d = -planeNormal.dot(Vec3d(0, 0, 0));

  planeNormalNorm = sqrt(planeNormal.squaredNorm());
  for (Particle &p : points) {
    p.normal = planeNormal;
  }
}

bool Plane::sphereIsInContact(const Vec3d &center_prim,
                              const Vec3d &sphereCenter, double sphereRadius) {
  double COLLISION_EPSILON = 0.2;
  Vec3d posShifted = sphereCenter - center_prim;
  double distToPlane = distanceToPlane(posShifted);
  double dist = std::abs(distToPlane);

  if (std::abs(distToPlane) > COLLISION_EPSILON + sphereRadius)
    return false;

  Vec3d p_prime = projectionOnPlane(planeNormal, d, posShifted);
  if (pointInsideTriangle(mesh[0], p_prime).first ||
      pointInsideTriangle(mesh[1], p_prime).first) {
    return true;
  }
  return false;
}

bool Plane::isInContact(const Vec3d &center_prim, const Vec3d &pos,
                        const Vec3d &velocity, Vec3d &normal, double &dist,
                        Vec3d &v_out) {

  Vec3d posShifted = pos - center_prim;
  double COLLISION_EPSILON = 0.05;
  double distToCenter = (posShifted).norm();

  if (distToCenter > boundaryRadius + COLLISION_EPSILON)
    return false;

  double distToPlane = distanceToPlane(posShifted);
  dist = distToPlane;

  if (std::abs(distToPlane) > COLLISION_EPSILON)
    return false;

  if ((distToPlane < 0) && (distToPlane * -1 > COLLISION_EPSILON + thickness))
    return false;

  Vec3d p_prime = projectionOnPlane(planeNormal, d, posShifted);

  // test if inside the plane
  if (pointInsideTriangle(mesh[0], p_prime).first ||
      pointInsideTriangle(mesh[1], p_prime).first) {
    normal = (distToPlane < -COLLISION_EPSILON ? -1 : 1) * planeNormal;
    v_out = this->velocity;
    return true;
  }

  // test if near edge
  std::vector<std::pair<Vec3d, Vec3d>> edges = {
      std::make_pair(upperLeft, upperRight),
      std::make_pair(upperRight, lowerRight),
      std::make_pair(lowerLeft, lowerRight),
      std::make_pair(upperLeft, lowerLeft)};

  double edgeTol = 0.0005;
  for (std::pair<Vec3d, Vec3d> &edge : edges) {
    std::pair<Vec3d, double> proj =
        projectionOnLine(edge.first, edge.second, posShifted);
    double t = proj.second;
    if ((posShifted - proj.first).norm() < edgeTol) {
      if ((t > -edgeTol) && (t < 1 + edgeTol)) {
        if (t < 0) {
          // collide with point A
          normal = (posShifted - edge.first).normalized();
        } else if (t > 1) {
          // collide with point B
          normal = (posShifted - edge.second).normalized();
        } else {
          // collide with edge
          normal = (posShifted - proj.first).normalized();
        }
        v_out = this->velocity;
        return true;
      }
    }
  }
  return false;
};

bool Plane::isInContact(const Vec3d &center_prim, const Vec3d &pos,
                        const Vec3d &velocity, Vec3d &normal, double &dist,
                        Vec3d &v_out, Vec3d &contact_point) {

  Vec3d posShifted = pos - center_prim;
  double COLLISION_EPSILON = 0.05;
  double distToCenter = (posShifted).norm();

  if (distToCenter > boundaryRadius + COLLISION_EPSILON)
    return false;

  double distToPlane = distanceToPlane(posShifted);
  dist = distToPlane;

  if (std::abs(distToPlane) > COLLISION_EPSILON)
    return false;

  if ((distToPlane < 0) && (distToPlane * -1 > COLLISION_EPSILON + thickness))
    return false;

  Vec3d p_prime = projectionOnPlane(planeNormal, d, posShifted);

  // test if inside the plane
  if (pointInsideTriangle(mesh[0], p_prime).first ||
      pointInsideTriangle(mesh[1], p_prime).first) {
    normal = (distToPlane < -COLLISION_EPSILON ? -1 : 1) * planeNormal;
    v_out = this->velocity;
    return true;
  }

  // test if near edge
  std::vector<std::pair<Vec3d, Vec3d>> edges = {
      std::make_pair(upperLeft, upperRight),
      std::make_pair(upperRight, lowerRight),
      std::make_pair(lowerLeft, lowerRight),
      std::make_pair(upperLeft, lowerLeft)};

  double edgeTol = 0.0005;
  for (std::pair<Vec3d, Vec3d> &edge : edges) {
    std::pair<Vec3d, double> proj =
        projectionOnLine(edge.first, edge.second, posShifted);
    double t = proj.second;
    if ((posShifted - proj.first).norm() < edgeTol) {
      if ((t > -edgeTol) && (t < 1 + edgeTol)) {
        if (t < 0) {
          // collide with point A
          normal = (posShifted - edge.first).normalized();
        } else if (t > 1) {
          // collide with point B
          normal = (posShifted - edge.second).normalized();
        } else {
          // collide with edge
          normal = (posShifted - proj.first).normalized();
        }
        v_out = this->velocity;
        return true;
      }
    }
  }
  return false;
};

Hanger2::Hanger2(Vec3d center, double scale, const char *filename, Vec3d color)
    : Primitive(HANGER2, center, false, color) {
  std::vector<Vec3d> posVec;
  std::vector<Vec3i> triVec;
  MeshFileHandler::loadOBJFile(filename, posVec, triVec);

  fcl::test::loadOBJFile(filename, vertices, triangles);

  for (int i = 0; i < posVec.size(); i++) {
    Vec3d point = Vec3d(posVec[i][0] * scale, posVec[i][1] * scale,
                        posVec[i][2] * scale) +
                  center;
    points.emplace_back(1, point, point, Vec3d(0, 0, 0), Vec2i(0, 0), i);
    vertices[i] = point;
  }
  for (int i = 0; i < triVec.size(); i++) {
    mesh.emplace_back(triVec[i][0], triVec[i][1], triVec[i][2], points);
    mesh[i].overrideColor = true;
    mesh[i].color = color;
  }

  for (Particle &p : points) {
    p.normal.setZero();
  }
  for (Triangle &t : mesh) {
    t.normal = t.getNormal(t.p0()->pos, t.p1()->pos, t.p2()->pos);
    t.p0()->normal += t.normal;
    t.p1()->normal += t.normal;
    t.p2()->normal += t.normal;
  }
  for (Particle &p : points) {
    p.normal.normalize();
  }

  model = std::make_shared<Model>();

  model->beginModel();
  model->addSubModel(vertices, triangles);
  model->endModel();
}

bool Hanger2::isInContact(const Vec3d &center_prim, const Vec3d &pos,
                          const Vec3d &velocity, Vec3d &normal, double &dist,
                          Vec3d &v_out, Vec3d &contact_point) {
  if (!isInCollision) {
    return false;
  }

  double COLLISION_EPSILON = parse_config2()[0];

  fcl::Transform3d pose1 = fcl::Transform3d::Identity();
  fcl::Transform3d pose2 = fcl::Transform3d::Identity();
  pose1.linear() = rotation_matrix;
  pose2.translation() = pos - center;

  fcl::CollisionObjectd *obj1 = new fcl::CollisionObjectd(model, pose1);
  std::shared_ptr<fcl::Sphered> sphere =
      std::make_shared<fcl::Sphered>(COLLISION_EPSILON);
  fcl::CollisionObjectd *obj2 = new fcl::CollisionObjectd(sphere, pose2);

  fcl::CollisionRequestd request(num_max_contacts, true);
  fcl::CollisionResultd result;

  int ret = fcl::collide(obj1, obj2, request, result);

  // Vec3d contact_position = Vec3d(0, 0, 0);

  if (ret) {
    std::vector<fcl::Contactd> contacts;
    result.getContacts(contacts);
    for (int i = 0; i < contacts.size(); i++) {
      // contact_position += contacts[i].pos;
      normal += contacts[i].normal;
    }
    normal.normalize();

    // contact_position = contact_position / contacts.size();
    // Vec3d r = contact_position - center;
    contact_point = pos;
    Vec3d v_T = angularVelocity.cross(contact_point - center);

    // contact_point = contact_position;
    v_out = this->velocity + v_T;

    return true;
  }
  return false;
}

Hanger::Hanger(Vec3d center, double w, double l, double h, double l2, double h2)
    : Primitive(HANGER, center, true, COLOR_WOOD) {
  primitives.reserve(12);

  Vec3d center2 = center + Vec3d(0., h + h2, 0.);

  left_plane = new Plane(center + Vec3d(0., 0., -w), center + Vec3d(l, h, -w),
                         center + Vec3d(-l, h, -w), COLOR_WOOD);
  right_plane = new Plane(center + Vec3d(0., 0., w), center + Vec3d(-l, h, w),
                          center + Vec3d(l, h, w), COLOR_WOOD);

  top_plane = new Plane(center + Vec3d(0., h, 0.), center + Vec3d(l, h, -w),
                        center + Vec3d(l, h, w), COLOR_WOOD);
  bottom_plane = new Plane(center + Vec3d(0., -h, 0.), center + Vec3d(l, -h, w),
                           center + Vec3d(l, -h, -w), COLOR_WOOD);

  front_plane = new Plane(center + Vec3d(-l, 0., 0.), center + Vec3d(-l, h, -w),
                          center + Vec3d(-l, h, w), COLOR_WOOD);
  back_plane = new Plane(center + Vec3d(l, 0., 0.), center + Vec3d(l, h, w),
                         center + Vec3d(l, h, -w), COLOR_WOOD);

  left2_plane =
      new Plane(center2 + Vec3d(0., 0., -w), center2 + Vec3d(l2, h2, -w),
                center2 + Vec3d(-l2, h2, -w), COLOR_WOOD);
  right2_plane =
      new Plane(center2 + Vec3d(0., 0., w), center2 + Vec3d(-l2, h2, w),
                center2 + Vec3d(l2, h2, w), COLOR_WOOD);

  top2_plane =
      new Plane(center2 + Vec3d(0., h2, 0.), center2 + Vec3d(l2, h2, -w),
                center2 + Vec3d(l2, h2, w), COLOR_WOOD);

  front2_plane =
      new Plane(center2 + Vec3d(-l2, 0., 0.), center2 + Vec3d(-l2, h2, -w),
                center2 + Vec3d(-l2, h2, w), COLOR_WOOD);
  back2_plane =
      new Plane(center2 + Vec3d(l2, 0., 0.), center2 + Vec3d(l2, h2, w),
                center2 + Vec3d(l2, h2, -w), COLOR_WOOD);

  underground = new Plane(Vec3d(0, -1.5, 0), Vec3d(7, -1.5, 7),
                          Vec3d(-7, -1.5, 7), COLOR_GRAY50);

  primitives.emplace_back(underground);
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

bool Hanger::isInContact(const Vec3d &center_prim, const Vec3d &pos,
                         const Vec3d &velocity, Vec3d &normal, double &dist,
                         Vec3d &v_out) {
  for (Primitive *p : primitives) {
    if (p->isInContact(center_prim + p->centerInit, pos, velocity, normal, dist,
                       v_out))
      return true;
  }
  return false;
}

Cube::Cube(Vec3d center, double l, double h, double w, Vec3d color)
    : Primitive(CUBE, center, true, color) {
  primitives.reserve(6);

  left_plane = new Plane(center + Vec3d(0., 0., -w), center + Vec3d(l, h, -w),
                         center + Vec3d(-l, h, -w), color);
  right_plane = new Plane(center + Vec3d(0., 0., w), center + Vec3d(-l, h, w),
                          center + Vec3d(l, h, w), color);

  top_plane = new Plane(center + Vec3d(0., h, 0.), center + Vec3d(l, h, -w),
                        center + Vec3d(l, h, w), color);
  bottom_plane = new Plane(center + Vec3d(0., -h, 0.), center + Vec3d(l, -h, w),
                           center + Vec3d(l, -h, -w), color);

  front_plane = new Plane(center + Vec3d(-l, 0., 0.), center + Vec3d(-l, h, -w),
                          center + Vec3d(-l, h, w), color);
  back_plane = new Plane(center + Vec3d(l, 0., 0.), center + Vec3d(l, h, w),
                         center + Vec3d(l, h, -w), color);

  primitives.emplace_back(back_plane);
  primitives.emplace_back(front_plane);
  primitives.emplace_back(bottom_plane);
  primitives.emplace_back(top_plane);
  primitives.emplace_back(right_plane);
  primitives.emplace_back(left_plane);
}

bool Cube::isInContact(const Vec3d &center_prim, const Vec3d &pos,
                       const Vec3d &velocity, Vec3d &normal, double &dist,
                       Vec3d &v_out) {
  if (!isInCollision) {
    return false;
  }
  for (Primitive *p : primitives) {
    if (p->isInContact(center_prim + p->centerInit, pos, velocity, normal, dist,
                       v_out))
      return true;
  }
  return false;
}

bool Cube::isInContact(const Vec3d &center_prim, const Vec3d &pos,
                       const Vec3d &velocity, Vec3d &normal, double &dist,
                       Vec3d &v_out, Vec3d &contact_point) {
  if (!isInCollision) {
    return false;
  }
  for (Primitive *p : primitives) {
    if (p->isInContact(center_prim + p->centerInit, pos, velocity, normal, dist,
                       v_out))
      return true;
  }
  return false;
}

Sphere::Sphere(Vec3d c, double radius, Vec3d myColor, int resolution,
               bool discretized)
    : Primitive(SPHERE, c, false, myColor), radius(radius), rotates(false),
      discretized(discretized) {
  int numX, numY;
  numX = numY = resolution;
  double d_phi = 360.0 / numX;
  double d_theta = 180.0 / numY;
  std::vector<std::vector<int>> particleTriangleMap;

  Vec3d color = myColor;
  mesh.reserve(numX * numY * 5);
  points.reserve(numX * numY * 5);
  Vec3d pos = getSpherePos(radius, 0, 0);
  points.emplace_back(1, pos, pos, Vec3d(0, 0, 0), Vec2i(0, 0), 0);

  auto createTriangle = [&](int id0, int id1, int id2, Vec3d triColor) {
    mesh.emplace_back(id2, id1, id0, points);
    int triIdx = mesh.size() - 1;
    mesh[triIdx].overrideColor = true;
    mesh[triIdx].color = triColor;
  };
  for (int y = 1; y < numY; y++) {
    for (int x = 0; x < numX; x++) {
      double theta = d_theta * y;
      double phi = d_phi * x;
      int idx = y * numX + x;
      pos = getSpherePos(radius, phi, theta);
      assert(std::abs(pos.norm() - radius) < 0.001);
      points.emplace_back(1, pos, pos, Vec3d(0, 0, 0), Vec2i(0, 0),
                          points.size());

      int last = ((int)points.size()) - 1;
      assert(points.size() >= 1);
      if ((y > 1) && (phi > 0) &&
          (idx - numX - 1 >= 0)) { // connect point with previous row
        createTriangle(last, last - 1, last - numX, color);
        assert(last - numX - 1 >= 0);
        createTriangle(last - 1, last - numX - 1, last - numX, color);
      } else if ((y == 1) && (x > 0)) {
        assert(last - 1 >= 0);
        createTriangle(last, last - 1, 0,
                       Vec3d(0, 1, 0)); // connect point with row 0
        if (x == numX - 1) {
          createTriangle(1, last, 0, color); // connect point with row 0
        }
      }
    }

    if ((y > 0) && (((int)points.size()) - 1 - numX + 1 - numX >= 0)) {
      int last = ((int)points.size()) - 1;
      assert(points.size() >= 1);
      assert(last - numX + 1 - numX >= 0);
      createTriangle(last - numX + 1 - numX, last - numX + 1, last, color);
      createTriangle(last, last - numX, last - numX + 1 - numX, color);
    }
  }

  pos = getSpherePos(radius, 0, 180);
  points.emplace_back(1, pos, pos, Vec3d(0, 0, 0), Vec2i(0, 0), points.size());
  int lastPointIdx = points.size() - 1;
  createTriangle(lastPointIdx, lastPointIdx - 1, lastPointIdx - numX, color);
  for (int i = points.size() - numX; i < (int)lastPointIdx; i++) {
    createTriangle(lastPointIdx, i - 1, i, color);
  }

  for (Particle &p : points) {
    p.normal.setZero();
  }
  for (Triangle &t : mesh) {
    t.normal = t.getNormal(t.p0()->pos, t.p1()->pos, t.p2()->pos);
    t.p0()->normal += t.normal;
    t.p1()->normal += t.normal;
    t.p2()->normal += t.normal;
  }

  for (Particle &p : points) {
    p.normal.normalize();
  }
};

bool Sphere::isInContact(const Vec3d &center_prim, const Vec3d &pos,
                         const Vec3d &velocity, Vec3d &normal, double &dist,
                         Vec3d &v_out) {

  double COLLISION_EPSILON = 0.1;
  dist = (pos - center_prim).norm() - radius;
  normal = (pos - center_prim).normalized();
  bool collides = (dist < COLLISION_EPSILON);

  if (discretized && collides) {
    double minDist = 1000;
    Triangle &closestTriangle = mesh[0];
    int count = 0;
    Vec3d transformed = pos - center_prim;
    for (Triangle &t : mesh) {
      Vec3d thisNormal;
      std::pair<bool, Vec3d> projecitonResult =
          pointInsideTriangle(t, transformed);
      if (projecitonResult.first) {
        Vec3d proj = projecitonResult.second;
        double dist = (transformed - proj).norm();
        if (dist < radius) {
          count++;
          minDist = dist;
          normal = t.normal;
        }
      }
    }
  }
  v_out = this->velocity;
  if (rotates) {
    v_out += Vec3d(0, 1, 0).cross(normal) * 8;
  }
  return collides;
};

bool Sphere::isInContact(const Vec3d &center_prim, const Vec3d &pos,
                         const Vec3d &velocity, Vec3d &normal, double &dist,
                         Vec3d &v_out, Vec3d &contact_point) {
  double COLLISION_EPSILON = 5e-3;
  dist = (pos - center_prim).norm() - radius;
  normal = (pos - center_prim).normalized();
  bool collides = (dist < COLLISION_EPSILON);

  v_out = this->velocity;
  return collides;
}

Vec3d flipXY(Vec3d in) {
  Vec3d out;
  out[0] = in[1];
  out[1] = -in[2];

  out[2] = in[0];
  return out;
}

Bowl::Bowl(Vec3d c, double radius, Vec3d myColor, int resolution)
    : Primitive(SPHERE, c, false, myColor), radius(radius) {
  int numX, numY;
  numX = numY = resolution;
  double d_phi = 360.0 / numX;
  double d_theta = 90.0 / numY;
  std::vector<std::vector<int>> particleTriangleMap;

  Vec3d color = myColor;
  mesh.reserve(numX * numY * 5);
  points.reserve(numX * numY * 5);
  Vec3d pos = flipXY(getSpherePos(radius, 0, 0));
  points.emplace_back(1, pos, pos, Vec3d(0, 0, 0), Vec2i(0, 0), 0);

  auto createTriangle = [&](int id0, int id1, int id2, Vec3d triColor) {
    mesh.emplace_back(id2, id1, id0, points);
    int triIdx = mesh.size() - 1;
    mesh[triIdx].overrideColor = true;
    mesh[triIdx].color = triColor;
  };
  for (int y = 1; y < numY; y++) {
    for (int x = 0; x < numX; x++) {
      double theta = d_theta * y;
      double phi = d_phi * x;
      int idx = y * numX + x;
      pos = flipXY(getSpherePos(radius, phi, theta));
      assert(std::abs(pos.norm() - radius) < 0.001);
      points.emplace_back(1, pos, pos, Vec3d(0, 0, 0), Vec2i(0, 0),
                          points.size());

      int last = ((int)points.size()) - 1;
      assert(points.size() >= 1);
      if ((y > 1) && (phi > 0) &&
          (idx - numX - 1 >= 0)) { // connect point with previous row
        createTriangle(last, last - 1, last - numX, color);
        assert(last - numX - 1 >= 0);
        createTriangle(last - 1, last - numX - 1, last - numX, color);

      } else if ((y == 1) && (x > 0)) {
        assert(last - 1 >= 0);
        createTriangle(last, last - 1, 0,
                       Vec3d(0, 1, 0)); // connect point with row 0
        if (x == numX - 1) {
          createTriangle(1, last, 0, color); // connect point with row 0
        }
      }
    }

    if ((y > 0) && (((int)points.size()) - 1 - numX + 1 - numX >= 0)) {
      int last = ((int)points.size()) - 1;
      assert(points.size() >= 1);
      assert(last - numX + 1 - numX >= 0);
      createTriangle(last - numX + 1 - numX, last - numX + 1, last, color);
      createTriangle(last, last - numX, last - numX + 1 - numX, color);
    }
  }

  for (Particle &p : points) {
    p.normal.setZero();
  }

  for (Triangle &t : mesh) {
    t.normal = t.getNormal(t.p0()->pos, t.p1()->pos, t.p2()->pos);
    t.p0()->normal += t.normal;
    t.p1()->normal += t.normal;
    t.p2()->normal += t.normal;
  }

  for (Particle &p : points) {
    p.normal.normalize();
  }
};

bool Bowl::isInContact(const Vec3d &center_prim, const Vec3d &pos,
                       const Vec3d &velocity, Vec3d &normal, double &dist,
                       Vec3d &v_out) {
  double COLLISION_EPSILON = 0.005;
  dist = (pos - center_prim).norm() - radius;
  normal = (center_prim - pos).normalized();
  v_out = this->velocity;

  if (dist > COLLISION_EPSILON) { // not inside sphere
    return false;
  }

  if (pos[1] > center_prim[1]) { // not in lower half of sphere
    return false;
  }

  return ((pos - center_prim).norm() > radius - COLLISION_EPSILON);
};

LowerLeg::LowerLeg(Vec3d center, Vec3d axis, double legLength,
                   double footLength)
    : Primitive(LOWER_LEG, center, true, COLOR_WOOD) {
  // center is position of the foot
  // axis is the direction of the foot
  // foot is the root of the joint "combination"

  primitives.reserve(2);
  rotation.setIdentity();
  axis.normalize();
  double radius = 0.8;
  double toeLength = 0.8;

  foot = new Capsule(Vec3d(0, 0, 0), radius, footLength, Vec3d(0, 1, 0), axis,
                     nullptr, true, this, COLOR_WOOD);
  Vec3d legCenter = foot->rotationFromParent * Vec3d(0, foot->length, 0);

  leg = new Capsule(legCenter, radius, legLength, axis, Vec3d(0, 0.7, 0.3),
                    foot, false, nullptr, COLOR_WOOD);
  joint = new Sphere(foot->rotationFromParent * Vec3d(0, foot->length, 0),
                     radius + 0.05, COLOR_WOOD);

  primitives.emplace_back(joint);
  primitives.emplace_back(foot); // leftLeg l=5
  primitives.emplace_back(leg);  // foot l=4
}

bool LowerLeg::isInContact(const Vec3d &center_prim, const Vec3d &pos,
                           const Vec3d &velocity, Vec3d &normal, double &dist,
                           Vec3d &v_out) {
  for (Primitive *p : primitives) {
    if (p->isInContact(center_prim + p->centerInit, pos, velocity, normal, dist,
                       v_out))
      return true;
  }
  return false;
}

Foot::Foot(Vec3d center, Rotation accumRot, Vec3d axis, double toeLength,
           double footLength, Vec3d color)
    : Primitive(FOOT, center, true, color) {
  primitives.reserve(6);
  double radius = 0.8;
  footBase = new Capsule(Vec3d(0, 0, 0), radius, footLength, Vec3d(0, 1, 0),
                         axis, nullptr, true, this, color);

  primitives.emplace_back(footBase); // leftLeg l=5
}

bool Foot::isInContact(const Vec3d &center_prim, const Vec3d &pos,
                       const Vec3d &velocity, Vec3d &normal, double &dist,
                       Vec3d &v_out) {
  for (Primitive *p : primitives) {
    if (p->isInContact(center_prim + p->centerInit, pos, velocity, normal, dist,
                       v_out))
      return true;
  }

  return false;
}

// Capsule(Vec3d center, double radius, double length, Vec3d myColor = Vec3d(1,
// 0, 0), int resSphere = 40, int resBody = 40);

Capsule::Capsule(Vec3d anchorCenter, double radius, double length,
                 Vec3d parentAxis, Vec3d axis, Capsule *parent, bool isRoot,
                 Primitive *rootPrimitiveContainer, Vec3d myColor,
                 int resSphere, int resBody)
    : Primitive(CAPSULE, anchorCenter, false, myColor), radius(radius),
      length(length), parent(parent), isRoot(isRoot),
      rootPrimitiveContainer(rootPrimitiveContainer) {
  // the capsule has three parts, top cap --> half sphere with r=radius, body
  // --> cylinder with length,r=radius  bottom cap --> half sphere
  // first create axis aligned mesh

  axis.normalize();
  parentAxis.normalize();
  this->axis = axis;
  this->parentAxis = parentAxis;
  this->rotationFromParent.setIdentity();
  this->globalRotation.setIdentity();

  rotationFromParent.setIdentity();
  Vec3d initialDir(0, 1, 0);
  axis.normalize();
  initialDir.normalize();
  rotationFromParent = axisToRotation(axis, initialDir);

  this->globalAxis = rotationFromParent * parentAxis;
  globalRotation = axisToRotation(this->globalAxis, initialDir);

  auto getCapsulePos = [&](double radius, double phi, double theta,
                           bool withRotation = true) {
    double x =
        radius * glm::cos(glm::radians(phi)) * glm::sin(glm::radians(theta));
    double z =
        radius * glm::sin(glm::radians(phi)) * glm::sin(glm::radians(theta));
    double y = radius * glm::cos(glm::radians(180 - theta));
    if (theta > 90) { // corresponds to top cap
      y += length;
    }
    if (withRotation)
      return globalRotation * Vec3d(x, y, z);
    return Vec3d(x, y, z);
  };

  auto createTriangle = [&](int id0, int id1, int id2, Vec3d triColor) {
    mesh.emplace_back(id2, id1, id0, points);
    int triIdx = mesh.size() - 1;
    mesh[triIdx].overrideColor = true;
    mesh[triIdx].color = triColor;
  };
  auto getColor = [&](double phi, double theta) { return myColor; };

  int numX, numY;
  numX = numY = resSphere;
  double d_phi = 360.0 / numX;
  double d_theta = 180.0 / numY;
  std::vector<std::vector<int>> particleTriangleMap;

  Vec3d color = myColor;
  mesh.reserve(numX * numY * 5);
  points.reserve(numX * numY * 5);
  Vec3d pos = getCapsulePos(radius, 0, 0);
  points.emplace_back(1, pos, pos, Vec3d(0, 0, 0), Vec2i(0, 0), 0);

  for (int y = 1; y < numY; y++) {
    double theta = d_theta * y;

    for (int x = 0; x < numX; x++) {
      double phi = d_phi * x;
      int idx = y * numX + x;
      pos = getCapsulePos(radius, phi, theta);
      points.emplace_back(1, pos, pos, Vec3d(0, 0, 0), Vec2i(0, 0),
                          points.size());

      int last = ((int)points.size()) - 1;
      if ((y > 1) && (phi > 0) &&
          (idx - numX - 1 >= 0)) { // connect point with previous row
        createTriangle(last, last - 1, last - numX, color);
        createTriangle(last - 1, last - numX - 1, last - numX, color);

      } else if ((y == 1) && (x > 0)) {
        createTriangle(last, last - 1, 0, color); // connect point with row 0
        if (x == numX - 1)
          createTriangle(1, last, 0, color); // connect point with row 0
      }
    }

    if ((y > 0) && (((int)points.size()) - 1 - numX + 1 - numX >= 0)) {
      int last = ((int)points.size()) - 1;
      //      createTriangle(last, last - numX + 1, last - numX + 1 - numX,
      //      color);
      createTriangle(last - numX + 1 - numX, last - numX + 1, last, color);
      createTriangle(last, last - numX, last - numX + 1 - numX, color);
    }
  }

  pos = getCapsulePos(radius, 0, 180);
  points.emplace_back(1, pos, pos, Vec3d(0, 0, 0), Vec2i(0, 0), points.size());
  int lastPointIdx = points.size() - 1;
  createTriangle(lastPointIdx, lastPointIdx - 1, lastPointIdx - numX, color);

  for (int i = points.size() - numX; i < (int)lastPointIdx; i++) {
    createTriangle(lastPointIdx, i - 1, i, color);
  }

  for (Particle &p : points) {
    p.normal.setZero();
  }
  for (Triangle &t : mesh) {
    t.normal = t.getNormal(t.p0()->pos, t.p1()->pos, t.p2()->pos);

    t.p0()->normal += t.normal;
    t.p1()->normal += t.normal;
    t.p2()->normal += t.normal;
  }

  for (Particle &p : points) {
    p.normal.normalize();
  }
}

bool Capsule::isInContact(const Vec3d &center_prim, const Vec3d &pos,
                          const Vec3d &velocity, Vec3d &normal, double &dist,
                          Vec3d &v_out) {
  double delta = 0.1;

  // transform back to capsule space where capsule's bottom cap is positioned at
  // (0,0,0). We keep rotations, as otherwise numerical issues will come up.

  {
    Vec3d posLocal = (pos - center_prim);
    v_out = this->velocity;

    Vec3d bottomCapCenter = Vec3d(0, 0, 0);
    Vec3d topCapCenter = globalRotation * Vec3d(0, length, 0);
    std::pair<Vec3d, double> proj =
        projectionOnLine(bottomCapCenter, topCapCenter, posLocal);
    double t = proj.second;
    if ((t < 0 - (radius) / length) || (t > 1 + (radius) / length))
      return false;

    if (t < 0) { // bottom cap
      dist = posLocal.norm() - radius;
      normal = posLocal.normalized();
    } else if (t > 1) { // top cap
      dist = (posLocal - topCapCenter).norm() - (radius + 0.1);
      normal = (posLocal - topCapCenter).normalized();
    } else {
      dist = (posLocal - proj.first).norm() - (radius + 0.1);
      normal = (posLocal - proj.first).normalized();
    }

    // body
  }

  return (dist < delta);
}
