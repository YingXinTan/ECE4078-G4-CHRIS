// look for ARUCO till covariance small,
// then look for fruits, 'P' to predict, and 'N' to save prediction
// 's' to save before escaping
python operate.py --ip {IP ADDR} --port 8000

// transform map using robot pose, save map as ESTMAP.txt
python generate_estimated_map.py --tf 1

                        OR
                        
// no transformation, save map as ESTMAP.txt
python generate_estimated_map.py --tf 0

// x: -1.1; y: 1.1; rad: -45


// SLAM for ARUCO (Note which markers are good/bad)
// operate_SLAM.py

// generate ARUCO map (You will know how good SLAM is)
// generate_estimated_ARUCO.py

// load ARUCO map, freeze markers state, look for fruit
// operate_CV.py

// estimate fruit position, save them as targets.txt
// TargetPoseEst.py

// to know how good fruits position is
// CV_eval.py

// Append targets.txt to ESTMAP.txt
