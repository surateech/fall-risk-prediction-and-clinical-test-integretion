# Fall_Risk_Monitoring_and_Clinical_Test_Integretion
Algorithms for Fall Risk Monitoring and Mobility Assessment using Azure Kinect DK

This has 3 necessary functions in the system following by:F
1. Fall Risk Monitoring (Percentage of Fall Risk)
2. TUG Test
3. Stepping Test

**Description**
The crucial component, which is Fall Risk Monitoring, has 3 significant parameters consisting of downward velocity, centerline angle, and aspect ratio to measure fall risk percentage with logistic regression that uses Sigmoid function for final result.

For the mobility assessment, we choose 2 tests that are Time Up and Go and Fukuda Stepping tests.

Additionally, if you want to check accuracy from these tests, Kinovea will be ground truth when you need to compare each other.

Finally, we uses Tkinder for generating GUI of the system, which combines all functions into the one place.

**Recommendation**
1. Before using Azure Kinect DK, please you downloads Visual Studio C++ Build Tools 2022
2. To make sure that this camera can be utilized, you should open it through Azure Kinect Viewer v1.4.2
3. You should download necessary libraries consisting of opencv2, pyk4a, mediapipe, and PIL (to run with Tkinder)

**Additional Download Link**
1. Visual Studio 2026 (For installing VS C++ Build Tools 2022): https://visualstudio.microsoft.com/vs/
2. Azure Kinect Viewer v1.4.2: https://github.com/microsoft/azure-kinect-sensor-sdk/blob/develop/docs/usage.md
3. Kinovea: https://www.kinovea.org/
