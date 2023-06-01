length = 0.4
ratio = 1.5
width = length*ratio
length_num = 32
width_num = int(length_num*ratio)
print(length_num, width_num)
center = [0.022, -0.198, 0.1]

def write_v(f, v):
    f.write("v {:.5f} {:.5f} {:.5f}\n".format(v[0]+center[0], v[1]+center[1],  v[2]+center[2]))

def write_t(f, t):
    f.write("f "+str(t[0])+" "+str(t[1])+" "+str(t[2])+"\n")

def write_t2(f, t):
    f.write("f "+str(t[0])+" "+str(t[2])+" "+str(t[1])+"\n")


with open('flag3.obj', 'w') as f:
# vertex
    for w in range(width_num):
        for l in range(length_num):
            write_v(f, [-w*width/width_num, 0., l*length/length_num])

# face
    for w in range(width_num-1):
        for l in range(length_num-1):
            write_t(f, [w*length_num+l+2, w*length_num+l+1, (w+1)*length_num+l+1])
        for l in range(length_num-1):
            write_t(f, [(w+1)*length_num+l+1, (w+1)*length_num+l+2, w*length_num+l+2])