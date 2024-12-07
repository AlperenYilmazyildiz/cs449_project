world: {}

robot_base (world): { 
    Q: "t(0 0 0.18)", 
    shape:box, 
    mass:1, 
    size:[0.1, 0.4, 0.2], 
    color:[0 1 0],
    joint: transXYPhi,
    limits: [-10,10,-10,10,-4,4]
}

joint_1 (robot_base): {
    joint: transZ,
    pre: "T t(0 0 0)",
    limits: [0.1, 3.]
}

robot_leg_1 (joint_1): {
    Q: "t(0.35 0.1 -0.09)", 
    shape:box, 
    mass:1, 
    size:[0.8, 0.08, 0.02], 
    color:[0 1 0] 
}

joint_2 (robot_base): {
    joint: transZ,
    pre: "T t(0 0 0)", 
    limits: [0.1, 3.]  
}

robot_leg_2 (joint_2): {
    Q: "t(0.35 -0.1 -0.09)", 
    shape:box, 
    mass:1, 
    size:[0.8, 0.08, 0.02], 
    color:[0 1 0] 
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


