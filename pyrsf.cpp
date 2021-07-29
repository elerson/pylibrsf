#include "pyrsf.h"
#include <iostream>
#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace py = pybind11;


PyLibRSF::PyLibRSF(std::string alg){
 
  SolverOptions.minimizer_progress_to_stdout = false;
  SolverOptions.use_nonmonotonic_steps = true;
  SolverOptions.trust_region_strategy_type = ceres::TrustRegionStrategyType::DOGLEG;
  SolverOptions.dogleg_type = ceres::DoglegType::SUBSPACE_DOGLEG;
  SolverOptions.max_num_iterations = 1000;
  SolverOptions.num_threads = 1;
  SolverOptions.max_solver_time_in_seconds = 0.25;
  counter = 0;
  ParseErrorModel(alg);

}
void PyLibRSF::addStates(double Timestamp){
  Graph.addState(POSITION_STATE, libRSF::DataType::Point2, Timestamp);
  Graph.addState(ORIENTATION_STATE, libRSF::DataType::Angle, Timestamp);

}
void PyLibRSF::setInitialPose(double Timestamp, double x, double y, double tetha, std::vector<float> covariance){

  Graph.addState(POSITION_STATE, libRSF::DataType::Point2, Timestamp);
  Graph.addState(ORIENTATION_STATE, libRSF::DataType::Angle, Timestamp);

  ceres::Vector pose(2), yaw(1);
  pose << x, y;
  yaw << tetha;


  libRSF::Data prior_pose = libRSF::Data(libRSF::DataType::Error2, 0);
  libRSF::Data prior_yaw = libRSF::Data(libRSF::DataType::Error1, 0);


  prior_pose.setValue(libRSF::DataElement::Mean, pose);
  prior_yaw.setValue(libRSF::DataElement::Mean, yaw);

  //POSITION_STATE
  libRSF::StateList PoseList;
  PoseList.add(POSITION_STATE, Timestamp);
  
  libRSF::GaussianDiagonal<2> noisePrior2;
  ceres::Vector std2(2); std2 << covariance[0], covariance[1]; 
  noisePrior2.setStdDevDiagonal(std2);

  Graph.addFactor<libRSF::FactorType::Prior2>(PoseList, prior_pose, noisePrior2);



  //ORIENTATION_STATE
  libRSF::StateList YawList;
  YawList.add(ORIENTATION_STATE, Timestamp);
  
  libRSF::GaussianDiagonal<1> noisePrior1;
  ceres::Vector std1(1); std1 << covariance[2]; 
  noisePrior1.setStdDevDiagonal(std1);

  Graph.addFactor<libRSF::FactorType::PriorAngle>(YawList, prior_yaw, noisePrior1);

}


void PyLibRSF::addMeasurement(double Timestamp, std::vector<vector<double>> &positions, std::vector<double> &range, std::vector<double> &L, std::vector<double> &covariance){
  
  libRSF::StateList ListRange;
  libRSF::Data Range(libRSF::DataType::Range2, Timestamp);
  libRSF::GaussianDiagonal<1> NoiseRange;
  libRSF::CorrentropyDiagonal<1> CorrNoiseRange;


  ListRange.add(POSITION_STATE, Timestamp);
  ListRange.add(ORIENTATION_STATE, Timestamp);
 
  
  /** default error model */
  static libRSF::GaussianMixture<1> GMM((libRSF::Vector2() << 0, 0).finished(),
                                         (libRSF::Vector2() << 0.1, 1.0).finished(),
                                         (libRSF::Vector2() << 0.5, 0.5).finished());

  counter++;
  for( int i = 0; i < range.size(); ++i){
    ceres::Vector mean_v(1); mean_v << range[i];
    ceres::Vector pos_v(3); pos_v << positions[i][0], positions[i][1], 0;
    ceres::Vector cov_v(1); cov_v << covariance[i];
    ceres::Vector L_v(1); L_v << L[i];



    Range.setValue(libRSF::DataElement::Mean, mean_v);
    Range.setValue(libRSF::DataElement::SatPos, pos_v);
    Range.setValue(libRSF::DataElement::L, L_v);
    
    
    /** add factor */
    switch(Config.Ranging.ErrorModel.Type)
    {
      case libRSF::ErrorModelType::Gaussian:
        NoiseRange.setStdDevDiagonal(cov_v*sqrt(range.size()));
        Graph.addFactor<libRSF::FactorType::Range2>(ListRange, Range, NoiseRange);
        break;
      
      case libRSF::ErrorModelType::Corr:
        {
          if(counter > 20){
            CorrNoiseRange.setParams(0, 1);
            CorrNoiseRange.setStdDevDiagonal(cov_v*sqrt(range.size()));
            
            Graph.addFactor<libRSF::FactorType::Range2>(ListRange, Range, CorrNoiseRange);
          }else{
            NoiseRange.setStdDevDiagonal(cov_v*sqrt(range.size()));
            Graph.addFactor<libRSF::FactorType::Range2>(ListRange, Range, NoiseRange);
          }
       }

        break;

      case libRSF::ErrorModelType::DCS:
        NoiseRange.setStdDevDiagonal(cov_v);
        Graph.addFactor<libRSF::FactorType::Range2>(ListRange, Range, NoiseRange, new libRSF::DCSLoss(1.0));
        break;

      case libRSF::ErrorModelType::cDCE:
        NoiseRange.setStdDevDiagonal(libRSF::Matrix11::Ones());
        Graph.addFactor<libRSF::FactorType::Range2>(ListRange, Range, NoiseRange, new libRSF::cDCELoss(covariance[i]));
        break;

      case libRSF::ErrorModelType::GMM:
        if (Config.Ranging.ErrorModel.MixtureType == libRSF::ErrorModelMixtureType::MaxMix)
        {
          static libRSF::MaxMix1 NoiseRangeMaxMix(GMM);
          Graph.addFactor<libRSF::FactorType::Range2>(ListRange, Range, NoiseRangeMaxMix);
        }
        else if (Config.Ranging.ErrorModel.MixtureType == libRSF::ErrorModelMixtureType::SumMix)
        {
          static libRSF::SumMix1 NoiseRangeSumMix(GMM);
          Graph.addFactor<libRSF::FactorType::Range2>(ListRange, Range, NoiseRangeSumMix);
        }
        else
        {
          PRINT_ERROR("Wrong error model mixture type!");
        }
        break;


      default:
        PRINT_ERROR("Wrong error model type: ", Config.Ranging.ErrorModel.Type);
        break;
    }
  
  }
}


void PyLibRSF::addOdometry(double Timestamp, double TimestampOld, vector<double> &mean, double wheelbase, vector<double> &covariance){

    /** add motion model or odometry */
    libRSF::StateList MotionList;
    MotionList.add(POSITION_STATE, TimestampOld);
    MotionList.add(ORIENTATION_STATE, TimestampOld);
    MotionList.add(POSITION_STATE, Timestamp);
    MotionList.add(ORIENTATION_STATE, Timestamp);
    
    libRSF::Data data(libRSF::DataType::Odom2Diff, 0);
    ceres::Vector mean_v(3); mean_v << mean[0], mean[1], mean[2];
    data.setValue(libRSF::DataElement::Mean, mean_v);
    ceres::Vector wheelbase_v(1); wheelbase_v << wheelbase;
    data.setValue(libRSF::DataElement::WheelBase, wheelbase_v);
    
    libRSF::GaussianDiagonal<3> NoiseOdom2Diff;
    ceres::Vector cov_v(3); cov_v << covariance[0], covariance[1], covariance[2];
    NoiseOdom2Diff.setStdDevDiagonal(cov_v);
    
    Graph.addFactor<libRSF::FactorType::Odom2Diff>(MotionList, data, NoiseOdom2Diff);
}


void PyLibRSF::tuneErrorModel()
{
  int NumberOfComponents = 2;
  if(Config.Ranging.ErrorModel.TuningType == libRSF::ErrorModelTuningType::EM)
  {
    std::vector<double> ErrorData;

    libRSF::GaussianMixture<1> GMM;

    /** fill empty GMM */
    if(GMM.getNumberOfComponents() == 0)
    {
      libRSF::GaussianComponent<1> Component;

      for(int nComponent = 0; nComponent < NumberOfComponents; ++nComponent)
      {

        Component.setParamsStdDev((libRSF::Vector1() << 0.1).finished()*pow(10, nComponent),
                                  (libRSF::Vector1() << 0.0).finished(),
                                  (libRSF::Vector1() << 1.0/NumberOfComponents).finished());

        GMM.addComponent(Component);
      }
    }

    Graph.computeUnweightedError(libRSF::FactorType::Range2, ErrorData);

    /** call the EM algorithm */
    libRSF::GaussianMixture<1>::EstimationConfig GMMConfig;
    GMMConfig.EstimationAlgorithm = libRSF::ErrorModelTuningType::EM;
    GMM.estimate(ErrorData, GMMConfig);

    /** apply error model */
    if(Config.Ranging.ErrorModel.MixtureType == libRSF::ErrorModelMixtureType::SumMix)
    {
      libRSF::SumMix1 NewSMModel(GMM);
      Graph.setNewErrorModel(libRSF::FactorType::Range2, NewSMModel);
    }
    else if(Config.Ranging.ErrorModel.MixtureType == libRSF::ErrorModelMixtureType::MaxMix)
    {
      libRSF::MaxMix1 NewMMModel(GMM);
      Graph.setNewErrorModel(libRSF::FactorType::Range2, NewMMModel);
    }
  }
}



bool PyLibRSF::ParseErrorModel(const std::string &ErrorModel)
{
  if(ErrorModel.compare("gauss") == 0)
  {
    Config.Ranging.ErrorModel.Type = libRSF::ErrorModelType::Gaussian;
    Config.Ranging.ErrorModel.TuningType = libRSF::ErrorModelTuningType::None;
  }
  else if(ErrorModel.compare("corr") == 0){
    Config.Ranging.ErrorModel.Type = libRSF::ErrorModelType::Corr;
    Config.Ranging.ErrorModel.TuningType = libRSF::ErrorModelTuningType::None;
  }
  else if(ErrorModel.compare("dcs") == 0)
  {
    Config.Ranging.ErrorModel.Type = libRSF::ErrorModelType::DCS;
    Config.Ranging.ErrorModel.TuningType = libRSF::ErrorModelTuningType::None;
  }
  else if(ErrorModel.compare("cdce") == 0)
  {
    Config.Ranging.ErrorModel.Type = libRSF::ErrorModelType::cDCE;
    Config.Ranging.ErrorModel.TuningType = libRSF::ErrorModelTuningType::None;
  }
  else if(ErrorModel.compare("sm") == 0)
  {
    Config.Ranging.ErrorModel.Type = libRSF::ErrorModelType::GMM;
    Config.Ranging.ErrorModel.MixtureType = libRSF::ErrorModelMixtureType::SumMix;
    Config.Ranging.ErrorModel.TuningType = libRSF::ErrorModelTuningType::None;
  }
  else if(ErrorModel.compare("mm") == 0)
  {
    Config.Ranging.ErrorModel.Type = libRSF::ErrorModelType::GMM;
    Config.Ranging.ErrorModel.MixtureType = libRSF::ErrorModelMixtureType::MaxMix;
    Config.Ranging.ErrorModel.TuningType = libRSF::ErrorModelTuningType::None;
  }
  else if(ErrorModel.compare("stsm") == 0)
  {
    Config.Ranging.ErrorModel.Type = libRSF::ErrorModelType::GMM;
    Config.Ranging.ErrorModel.MixtureType = libRSF::ErrorModelMixtureType::SumMix;
    Config.Ranging.ErrorModel.TuningType = libRSF::ErrorModelTuningType::EM;
  }
  else if(ErrorModel.compare("stmm") == 0)
  {
    Config.Ranging.ErrorModel.Type = libRSF::ErrorModelType::GMM;
    Config.Ranging.ErrorModel.MixtureType = libRSF::ErrorModelMixtureType::MaxMix;
    Config.Ranging.ErrorModel.TuningType = libRSF::ErrorModelTuningType::EM;
  }
  else
  {
    PRINT_ERROR("Wrong Error Model: ", ErrorModel);
    return false;
  }

  return true;
}

int PyLibRSF::teste(vector<double> &mean){
  return mean[0]*mean[1];
}

void PyLibRSF::solve(double Timestamp, double Window){

  Graph.solve(SolverOptions);
   
  /** apply sliding window */
  Graph.marginalizeAllStatesOutsideWindow(Window, Timestamp);
  Graph.removeAllFactorsOutsideWindow(Window, Timestamp);
}

std::vector<double> PyLibRSF::getState(double Timestamp){
  auto state_position = Graph.getStateData().getElement(POSITION_STATE, Timestamp, 0).getMean();
  auto state_orientation = Graph.getStateData().getElement(ORIENTATION_STATE, Timestamp, 0).getMean();

  return {state_position[0], state_position[1], state_orientation[0]};
}
    


//PYBIND11_MAKE_OPAQUE(std::string);

PYBIND11_MODULE(pylibrsf, m) {
   py::class_<PyLibRSF>(m, "PyLibRSF")
      .def(py::init<std::string>())
      .def("setInitialPose", &PyLibRSF::setInitialPose)
      .def("addStates", &PyLibRSF::addStates)  
      .def("addOdometry", &PyLibRSF::addOdometry)  
      .def("addMeasurement", &PyLibRSF::addMeasurement)  
      .def("tuneErrorModel", &PyLibRSF::tuneErrorModel)  
      .def("solve", &PyLibRSF::solve)
      .def("getState", &PyLibRSF::getState)
      .def("teste", &PyLibRSF::teste);    
}  




