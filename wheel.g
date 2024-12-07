base: { shape: cylinder, size: [.03, .01], color: [0, 0, 0] }
(base): { Q: "t(0 0 .02)", shape: cylinder, size: [.03, .02], color: [.3, .3, .3] }
steerJoint(base): {joint: hingeZ, shape: marker, size: [.1], ctrl_H: .1, limits: [0, .1] }
center(steerJoint): { Q: "t(-.02 .0 -.03)",  shape: marker, size: [.1] }
wheelJoint(center): { joint: hingeY, ctrl_H: .001, limits: [0, .1] }

(wheelJoint): { shape: ssBoxElip, size: [.0, .01, .0, .02, .01, .02, 0], color: [.3, .3, .3] }

