#ifndef PYGEOSTEINER_H
#define PYGEOSTEINER_H

#include <vector>
#include <string>

#include "libRSF.h"

#define POSITION_STATE "Position"
#define ORIENTATION_STATE "Orientation"

class PyLibRSF{
  public:

    PyLibRSF(std::string alg);
    void setInitialPose(double Timestamp, double x, double y, double tetha, std::vector<float> covariance);

    void addStates(double Timestamp);
    void addOdometry(double Timestamp, double TimestampOld, vector<double> &mean, double wheelbase, vector<double> &covariance);
    void addMeasurement(double Timestamp, std::vector<vector<double>> &positions, std::vector<double> &range, std::vector<double> &L, std::vector<double> &covariance);
    void tuneErrorModel();
    bool ParseErrorModel(const std::string &ErrorModel);
    void solve(double Timestamp, double Window);
    std::vector<double> getState(double Timestamp);
    int teste(vector<double> &mean);
    

  private:
    ceres::Solver::Options SolverOptions;
    libRSF::FactorGraph Graph;
    libRSF::FactorGraphConfig Config;
    int counter;
};

 

   
#endif
