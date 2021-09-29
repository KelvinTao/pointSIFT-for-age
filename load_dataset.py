import pickle
import os
import numpy as np

def load_data_npz(npz_path):
    d = np.load(npz_path)
    return d["image"], d["age"], d["img_size"]

def calweights(c_age)
    ##input: c_age, Nsample x Nage_thresh x 2 ,catrgorical age
    ##output: 
    ##sample_weights: calculate sample weights in c_age set
    ##Nsample,age for each sample
    labeln=np.sum(c_age[:,:,0],axis=1)
    ##age distribution, age_i: index1, index2 
    ##[[index1,index2,...],[index7,index10,...],[]];Nage
    labes=list(range(min(labeln),max(labeln)+1))
    mydict=list(map(lambda x:[i for i,j in enumerate(labeln) if j==x],labes))
    ##age hist: age_i: Num_age_i
    lenth=list(map(lambda x:len(x),mydict))
    lenths=list(np.asarray(lenth)[np.asarray(lenth)>0])
    ##average num at one age
    assert c_age.shape[0]==sum(lenth)
    #assert len(c_age)==sum(lenth)
    meanle=sum(lenth)//len(lenths)
    ##Nsample
    sample_weights=np.ones(len(labeln))
    for i in range(len(mydict)):
        if lenth[i]>0:
            sample_weights[mydict[i]]=min(round(meanle/lenth[i],2),25)
    return sample_weights

    '''
    def load_data(self)
        ##load train and test; 
        ##data: Nsample x Npoints x Nchannel; channel: X Y Z (+ R G B)
        ##label: Nsample x N_age_thresh x 2: age_thresh:0,1,2,...,99; [>,<]age_thresh
        train_images,train_labels,imgsize=load_data_npz(DATA_PATH+'/'+npz+'.train.npz')
        val_images,val_labels,imgsize=load_data_npz(DATA_PATH+'/'+npz+'.test.npz')
        print('Data loaded.')
        ##train$$#m_image.pop(numtest-1)#m_image=np.concatenate((m_image[0],m_image[1],m_image[2],m_image[3]), axis=0)
        ##train sample weights
        costweight=calweights(train_labels)
        ##test sample weights
        maeweight=calweights(val_labels)
        #training logits length
        NUM_CLASSES =int(val_labels.shape[1]*2)
    '''
    
           


def load_data(path,npz)
    ##load train and test; 
    ##data: Nsample x Npoints x Nchannel; channel: X Y Z (+ R G B)
    ##label: Nsample x N_age_thresh x 2: age_thresh:0,1,2,...,99; [>,<]age_thresh
    train_images,train_labels,imgsize=load_data_npz(path+'/'+npz+'.train.npz')
    val_images,val_labels,imgsize=load_data_npz(path+'/'+npz+'.test.npz')
    print('Data loaded.')
    ##train$$#m_image.pop(numtest-1)#m_image=np.concatenate((m_image[0],m_image[1],m_image[2],m_image[3]), axis=0)

    ##train sample weights
    costweight=calweights(train_labels)
    ##test sample weights
    maeweight=calweights(val_labels)
    #training logits length
    NUM_CLASSES =int(val_labels.shape[1]*2)



class ScannetDataset():
    def __init__(self, root, npoints=8192, split='train'):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_filename = os.path.join(self.root, 'scannet_%s.pickle' % split)
        with open(self.data_filename, 'rb') as fp:
            self.scene_points_list = pickle.load(fp, encoding='bytes')
            self.semantic_labels_list = pickle.load(fp, encoding='bytes')
        if split == 'train':
            labelweights = np.zeros(21)
            for seg in self.semantic_labels_list:
                tmp, _ = np.histogram(seg, range(22))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        elif split == 'test':
            self.labelweights = np.ones(21)

    def __getitem__(self, index):
        point_set = self.scene_points_list[index]
        semantic_seg = self.semantic_labels_list[index].astype(np.int32)
        coordmax = np.max(point_set,axis=0)
        coordmin = np.min(point_set,axis=0)
        smpmin = np.maximum(coordmax-[1.5, 1.5, 3.0], coordmin)
        smpmin[2] = coordmin[2]
        smpsz = np.minimum(coordmax-smpmin, [1.5, 1.5, 3.0])
        smpsz[2] = coordmax[2]-coordmin[2]
        cur_semantic_seg = None
        cur_point_set = None
        mask = None
        for i in range(10):
            curcenter = point_set[np.random.choice(len(semantic_seg),1)[0],:]
            curmin = curcenter-[0.75, 0.75, 1.5]
            curmax = curcenter+[0.75, 0.75, 1.5]
            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]
            curchoice = np.sum((point_set >= (curmin-0.2))*(point_set <= (curmax+0.2)),axis=1) == 3
            cur_point_set = point_set[curchoice, :]
            cur_semantic_seg = semantic_seg[curchoice]
            if len(cur_semantic_seg) == 0:
                continue
            mask = np.sum((cur_point_set >= (curmin-0.01))*(cur_point_set <= (curmax+0.01)), axis=1) == 3
            vidx = np.ceil((cur_point_set[mask, :]-curmin)/(curmax-curmin)*[31.0, 31.0, 62.0])
            vidx = np.unique(vidx[:, 0] * 31.0 * 62.0 + vidx[:, 1] * 62.0 + vidx[:, 2])
            isvalid = np.sum(cur_semantic_seg > 0)/len(cur_semantic_seg) >= 0.7 and len(vidx)/31.0/31.0/62.0 >= 0.02
            if isvalid:
                break
        choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
        point_set = cur_point_set[choice,:]

        semantic_seg = cur_semantic_seg[choice]
        mask = mask[choice]
        sample_weight = self.labelweights[semantic_seg]
        sample_weight *= mask
        return point_set, semantic_seg, sample_weight

    def __len__(self):
        return len(self.scene_points_list)


class ScannetDatasetWholeScene(object):
    def __init__(self, root, npoints=8192, split='train'):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_filename = os.path.join(self.root, 'scannet_%s.pickle' % split)
        with open(self.data_filename, 'rb') as fp:
            self.scene_points_list = pickle.load(fp, encoding='bytes')
            self.semantic_labels_list = pickle.load(fp, encoding='bytes')
        if split == 'train':
            labelweights = np.zeros(21)
            for seg in self.semantic_labels_list:
                tmp, _ = np.histogram(seg, range(22))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        elif split == 'test':
            self.labelweights = np.ones(21)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
        coordmax = np.max(point_set_ini, axis=0)
        coordmin = np.min(point_set_ini, axis=0)
        nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/1.5).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/1.5).astype(np.int32)
        nsubvolume_x *= 2
        nsubvolume_y *= 2
        point_sets = list()
        semantic_segs = list()
        sample_weights = list()
        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                curmin = coordmin+[i * 0.75, j * 0.75, 0]
                curmax = coordmin+[i * 0.75 + 1.5, j * 0.75 + 1.5, coordmax[2]-coordmin[2]]
                curchoice = np.sum((point_set_ini >= (curmin-0.2))*(point_set_ini <= (curmax+0.2)), axis=1) == 3
                cur_point_set = point_set_ini[curchoice, :]
                cur_semantic_seg = semantic_seg_ini[curchoice]
                if len(cur_semantic_seg) == 0:
                    continue
                mask = np.sum((cur_point_set >= (curmin-0.001))*(cur_point_set <= (curmax+0.001)), axis=1) == 3
                choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
                point_set = cur_point_set[choice, :]  # Nx3
                semantic_seg = cur_semantic_seg[choice]  # N
                mask = mask[choice]
                if sum(mask) < 2000:
                    continue
                if sum(mask)/float(len(mask)) < 0.01:
                    continue
                sample_weight = self.labelweights[semantic_seg]
                sample_weight *= mask  # N

                point_sets.append(np.expand_dims(point_set, 0))  # 1xNx3
                semantic_segs.append(np.expand_dims(semantic_seg, 0))  # 1xN
                sample_weights.append(np.expand_dims(sample_weight, 0))  # 1xN
        point_sets = np.concatenate(tuple(point_sets), axis=0)
        semantic_segs = np.concatenate(tuple(semantic_segs), axis=0)
        sample_weights = np.concatenate(tuple(sample_weights), axis=0)
        return point_sets, semantic_segs, sample_weights

    def __len__(self):
        return len(self.scene_points_list)




