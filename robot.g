world: {}

robot_base (world): {
    Q: "t(0 0 0)", 
    shape:box, 
    mass:0.1, 
    size:[0.1, 0.4, 0.01], 
    color:[0 0 0],
    joint: transXYPhi,
    limits: [-20,20,-20,20,-4,4],
    contact: 1
}

robot_body (robot_base): { 
    Q: "t(0 0 0.1)", 
    shape:box, 
    mass:0.1, 
    size:[0.1, 0.4, 0.2], 
    color:[0 1 0]
}

robot_marker (robot_body): {
    Q: "t(0 0 0.3)", 
    shape: marker,
    size: [.3]
}

joint_1 (robot_body): {
    joint: transZ,
    pre: "T t(0 0 0)",
    limits: [0.05, 0.1]
}

robot_leg_1 (joint_1): {
    Q: "t(0.35 0.1 -0.09)", 
    shape:box, 
    mass:1, 
    size:[0.8, 0.08, 0.02], 
    color:[0 1 0] 
}

joint_2 (robot_body): {
    joint: transZ,
    pre: "T t(0 0 0)", 
    limits: [0.05, 0.1]  
}

robot_leg_2 (joint_2): {
    Q: "t(0.35 -0.1 -0.09)", 
    shape:box, 
    mass:1, 
    size:[0.8, 0.08, 0.02], 
    color:[0 1 0] 
}

middle_joint(robot_body): {
    Q: "t(0.4 0 0)",
    shape: marker,
    size: [.1]
}

#Prefix: "W1_", Include: <wheel.g>
#Prefix: "W2_", Include: <wheel.g>
#Prefix: "W3_", Include: <wheel.g>
#Prefix: "W4_", Include: <wheel.g>
#Prefix!
 
#Edit W1_base (robot_base): { Q: "t(0 0.1 -0.12)" }
#Edit W2_base (robot_base): { Q: "t(0 -0.1 -0.12)" }
#Edit W3_base (robot_base): { Q: "t(0.6 0.1 -0.12)" }
#Edit W4_base (robot_base): { Q: "t(0.6 -0.1 -0.12)" }


