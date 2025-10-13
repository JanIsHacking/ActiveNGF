# This file is a modified version of https://github.com/mahaoxiang822/Generalizing-Grasp/blob/main/evaluate.py


from graspnetAPI import GraspGroup, GraspNetEval
import numpy as np
from graspnetAPI.graspnet_eval import *
from graspnetAPI.utils.eval_utils import *
import os
import torch


def eval_grasp(grasp_group, models, dexnet_models, poses, config, table=None, voxel_size=0.008, TOP_K=50):
    '''
    **Input:**

    - grasp_group: GraspGroup instance for evaluation.

    - models: in model coordinate

    - dexnet_models: models in dexnet format

    - poses: from model to camera coordinate

    - config: dexnet config.

    - table: in camera coordinate

    - voxel_size: float of the voxel size.

    - TOP_K: int of the number of top grasps to evaluate.
    '''
    num_models = len(models)
    ## grasp nms
    grasp_group = grasp_group.nms(0.03, 30.0 / 180 * np.pi)
    ## assign grasps to object
    # merge and sample scene
    model_trans_list = list()
    seg_mask = list()
    for i, model in enumerate(models):
        model_trans = transform_points(model, poses[i])
        seg = i * np.ones(model_trans.shape[0], dtype=np.int32)
        model_trans_list.append(model_trans)
        seg_mask.append(seg)
    seg_mask = np.concatenate(seg_mask, axis=0)
    scene = np.concatenate(model_trans_list, axis=0)

    # assign grasps
    indices = compute_closest_points(grasp_group.translations, scene)
    model_to_grasp = seg_mask[indices]
    pre_grasp_list = list()
    for i in range(num_models):
        grasp_i = grasp_group[model_to_grasp == i]
        grasp_i.sort_by_score()
        pre_grasp_list.append(grasp_i[:5].grasp_group_array)

    grasp_list = pre_grasp_list

    ## collision detection
    if table is not None:
        scene = np.concatenate([scene, table])

    collision_mask_list, empty_list, dexgrasp_list = collision_detection(
        grasp_list, model_trans_list, dexnet_models, poses, scene, outlier=0.05, return_dexgrasps=True)

    ## evaluate grasps
    # score configurations
    force_closure_quality_config = dict()
    fc_list = np.array([1.2, 1.0, 0.8, 0.6, 0.4, 0.2])
    for value_fc in fc_list:
        value_fc = round(value_fc, 2)
        config['metrics']['force_closure']['friction_coef'] = value_fc
        force_closure_quality_config[value_fc] = GraspQualityConfigFactory.create_config(
            config['metrics']['force_closure'])
    # get grasp scores
    score_list = list()

    for i in range(num_models):
        dexnet_model = dexnet_models[i]
        collision_mask = collision_mask_list[i]
        dexgrasps = dexgrasp_list[i]
        scores = list()
        num_grasps = len(dexgrasps)
        for grasp_id in range(num_grasps):

            if dexgrasps[grasp_id] is None:
                scores.append(-1.)
                continue
            if collision_mask[grasp_id]:
                scores.append(-1.)
                continue
            grasp = dexgrasps[grasp_id]
            score = get_grasp_score(grasp, dexnet_model, fc_list, force_closure_quality_config)
            scores.append(score)
        score_list.append(np.array(scores))

    return grasp_list, score_list, collision_mask_list

def get_scene_models(scene_id, ann_id, dataset_root, camera):
    '''
        return models in model coordinate
    '''
    model_dir = os.path.join(dataset_root, 'models')
    # print('Scene {}, {}'.format(scene_id, camera))
    scene_reader = xmlReader(os.path.join(dataset_root, 'scenes', get_scene_name(scene_id), camera, 'annotations', '%04d.xml' % (ann_id,)))
    posevectors = scene_reader.getposevectorlist()
    obj_list = []
    model_list = []
    dexmodel_list = []
    for posevector in posevectors:
        obj_idx, _ = parse_posevector(posevector)
        obj_list.append(obj_idx)
    for obj_idx in obj_list:
        model = o3d.io.read_point_cloud(os.path.join(model_dir, '%03d' % obj_idx, 'nontextured.ply'))
        dex_cache_path = os.path.join(dataset_root, 'dex_models', '%03d.pkl' % obj_idx)
        if os.path.exists(dex_cache_path):
            with open(dex_cache_path, 'rb') as f:
                dexmodel = pickle.load(f)
        else:
            dexmodel = load_dexnet_model(os.path.join(model_dir, '%03d' % obj_idx, 'textured'))
        points = np.array(model.points)
        model_list.append(points)
        dexmodel_list.append(dexmodel)
    return model_list, dexmodel_list, obj_list

def get_model_poses(scene_id, ann_id, dataset_root, camera):
    '''
    **Input:**

    - scene_id: int of the scen index.

    - ann_id: int of the annotation index.

    **Output:**

    - obj_list: list of int of object index.

    - pose_list: list of 4x4 matrices of object poses.

    - camera_pose: 4x4 matrix of the camera pose relative to the first frame.

    - align mat: 4x4 matrix of camera relative to the table.
    '''
    scene_dir = os.path.join(dataset_root, 'scenes')
    camera_poses_path = os.path.join(dataset_root, 'scenes', get_scene_name(scene_id), camera, 'camera_poses.npy')
    camera_poses = np.load(camera_poses_path)
    camera_pose = camera_poses[ann_id]
    align_mat_path = os.path.join(dataset_root, 'scenes', get_scene_name(scene_id), camera, 'cam0_wrt_table.npy')
    align_mat = np.load(align_mat_path)
    # print('Scene {}, {}'.format(scene_id, camera))
    scene_reader = xmlReader(os.path.join(scene_dir, get_scene_name(scene_id), camera, 'annotations', '%04d.xml'% (ann_id,)))
    posevectors = scene_reader.getposevectorlist()
    obj_list = []
    pose_list = []
    for posevector in posevectors:
        obj_idx, mat = parse_posevector(posevector)
        obj_list.append(obj_idx)
        pose_list.append(mat)
    return obj_list, pose_list, camera_pose, align_mat

def generate_scene(scene_id, dataset_root, camera):
    model_list, dexmodel_list, obj_list = get_scene_models(scene_id, 0, dataset_root, camera)
    model_sampled_list = list()
    for model in model_list:
        model_sampled = voxel_sample_points(model, 0.008)
        model_sampled_list.append(model_sampled)
    model_trans_list = list()
    _, pose_list, camera_pose, align_mat = get_model_poses(scene_id, 0, dataset_root, camera)
    # transform pose to world coord
    # note: because we use the view 0, only align_mat is needed
    transform_pose_list = []
    for pose in pose_list:
        pose = np.matmul(align_mat, pose)
        transform_pose_list.append(pose)

    seg_mask = list()
    for i, model in enumerate(model_sampled_list):
        model_trans = transform_points(model, transform_pose_list[i])
        seg = (obj_list[i]+1) * np.ones(model_trans.shape[0], dtype=np.int32)
        model_trans_list.append(model_trans)
        seg_mask.append(seg)
    seg_mask = np.concatenate(seg_mask, axis=0)
    scene = np.concatenate(model_trans_list, axis=0)
    return scene,seg_mask

class GraspNetEvalComplete(GraspNetEval):
    def eval_scene_objects(self, scene_id, dump_folder, dataset_root, camera, grasps_per_object=5, return_list=False, vis=False, max_width=0.1):
        config = get_config()
        table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008)

        list_coe_of_friction = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
        model_list, dexmodel_list, _ = get_scene_models(scene_id, 0, dataset_root, camera)
        model_sampled_list = list()
        for model in model_list:
            model_sampled = voxel_sample_points(model, 0.008)
            model_sampled_list.append(model_sampled)

        scene_accuracy = []
        grasp_list_list = []
        score_list_list = []
        collision_list_list = []

        grasp_group = GraspGroup().from_npy(
            os.path.join(dump_folder, get_scene_name(scene_id), self.camera, 'result.npy'))
        
        ####################
        # Retrieve the specified number of grasps per object
        # grasp_data = grasp_group.grasp_group_array
        # centers = grasp_data[:,8:11]
        # print("centers shape: ", centers.shape)
        # print("centers: ", centers)
        # print("grasp_data shape: ", grasp_data.shape)

        # scene, seg_mask_ = generate_scene(scene_id, dataset_root, camera)
        # indices = compute_closest_points(centers, scene)
        # model_to_grasp = seg_mask_[indices]
        # model_ids = np.unique(model_to_grasp)
        # print("model_ids: ", model_ids)
        # pred_list = []
        # for i in model_ids:
        #     if i==0:
        #         continue
        #     grasp_i = grasp_data[model_to_grasp == i]
        #     index = np.argsort(grasp_i[:,0])
        #     index = index[::-1]
        #     grasp_i = grasp_i[index]
        #     grasp_i[:,-1] = i
        #     pred_list.append(grasp_i[:grasps_per_object])
        # for x in pred_list:
        #     print("x shape: ", x.shape)
        # pred_list = np.concatenate(pred_list,axis=0)
        # approach = pred_list[:,4:7]
        # grasp_angle = pred_list[:,7]
        # grasp_center = pred_list[:,8:11]
        # obj_ids = pred_list[:,11,None]

        # rotation_matrix = batch_viewpoint_params_to_matrix(approach, grasp_angle)
        # rotation_matrix = rotation_matrix.reshape(-1, 9)

        # gg_array = np.concatenate([pred_list[:,0:4], rotation_matrix, grasp_center, obj_ids],axis=-1)
        # print("gg_array shape: ", gg_array.shape)
        
        # grasp_group = GraspGroup(gg_array)
        # gg_array = grasp_group.grasp_group_array
        model_ids = list(range(50))

        ####################

        _, pose_list, camera_pose, align_mat = self.get_model_poses(scene_id, 0)
        # transform pose to world coord
        # note: because we use the view 0, only align_mat is needed
        transform_pose_list = []
        for pose in pose_list:
            pose = np.matmul(align_mat, pose)
            transform_pose_list.append(pose)

        gg_array = grasp_group.grasp_group_array
        min_width_mask = (gg_array[:, 1] < 0)
        max_width_mask = (gg_array[:, 1] > max_width)
        gg_array[min_width_mask, 1] = 0
        gg_array[max_width_mask, 1] = max_width
        grasp_group.grasp_group_array = gg_array
        grasp_list, score_list, collision_mask_list = eval_grasp(grasp_group, model_sampled_list, dexmodel_list,
                                                                 transform_pose_list, config, table=table,
                                                                 voxel_size=0.008)
        # remove empty
        grasp_list = [x for x in grasp_list if len(x) != 0]
        score_list = [x for x in score_list if len(x) != 0]
        collision_mask_list = [x for x in collision_mask_list if len(x) != 0]

        if len(grasp_list) == 0:
            grasp_accuracy = np.zeros((grasps_per_object*len(model_ids), len(list_coe_of_friction)))
            scene_accuracy.append(grasp_accuracy)
            grasp_list_list.append([])
            score_list_list.append([])
            collision_list_list.append([])
            # print('\rMean Accuracy for scene:{}'.format(scene_id), np.mean(grasp_accuracy[:, :]),
            #       end='')

        # concat into scene level
        grasp_list, score_list, collision_mask_list = np.concatenate(grasp_list), np.concatenate(
            score_list), np.concatenate(collision_mask_list)
        # sort in scene level
        grasp_confidence = grasp_list[:, 0]
        indices = np.argsort(-grasp_confidence)
        grasp_list, score_list, collision_mask_list = grasp_list[indices], score_list[indices], collision_mask_list[
            indices]

        grasp_accuracy = np.zeros((grasps_per_object*len(model_ids), len(list_coe_of_friction)))
        for fric_idx, fric in enumerate(list_coe_of_friction):
            for k in range(0, grasps_per_object*len(model_ids)):
                if k + 1 > len(score_list):
                    grasp_accuracy[k, fric_idx] = np.sum(((score_list <= fric) & (score_list > 0)).astype(int)) / (
                            k + 1)
                else:
                    grasp_accuracy[k, fric_idx] = np.sum(
                        ((score_list[0:k + 1] <= fric) & (score_list[0:k + 1] > 0)).astype(int)) / (k + 1)

        # print('\rMean Accuracy for scene:%04d = %.3f' % (
        #     scene_id, 100.0 * np.mean(grasp_accuracy[:, :])), end='', flush=True)
        scene_accuracy.append(np.mean(grasp_accuracy,axis=0,keepdims=True))
        if not return_list:
            return scene_accuracy
        else:
            return scene_accuracy, grasp_list_list, score_list_list, collision_list_list

    def eval_scene(self, scene_id, dump_folder, TOP_K=50, return_list=False, vis=False, max_width=0.1):
        '''
        **Input:**

        - scene_id: int of the scene index.

        - dump_folder: string of the folder that saves the dumped npy files.

        - TOP_K: int of the top number of grasp to evaluate

        - return_list: bool of whether to return the result list.

        - vis: bool of whether to show the result

        - max_width: float of the maximum gripper width in evaluation

        **Output:**

        - scene_accuracy: np.array of shape (256, 50, 6) of the accuracy tensor.
        '''
        config = get_config()
        table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008)

        list_coe_of_friction = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
        model_list, dexmodel_list, _ = self.get_scene_models(scene_id, ann_id=0)
        num_model = len(model_list)
        TOP_K = num_model*5 # 50
        model_sampled_list = list()
        for model in model_list:
            model_sampled = voxel_sample_points(model, 0.008)
            model_sampled_list.append(model_sampled)

        scene_accuracy = []
        grasp_list_list = []
        score_list_list = []
        collision_list_list = []

        grasp_group = GraspGroup().from_npy(
            os.path.join(dump_folder, get_scene_name(scene_id), self.camera, 'result.npy'))

        _, pose_list, camera_pose, align_mat = self.get_model_poses(scene_id, 0)
        # transform pose to world coord
        # note: because we use the view 0, only align_mat is needed
        transform_pose_list = []
        for pose in pose_list:
            pose = np.matmul(align_mat, pose)
            transform_pose_list.append(pose)

        gg_array = grasp_group.grasp_group_array
        min_width_mask = (gg_array[:, 1] < 0)
        max_width_mask = (gg_array[:, 1] > max_width)
        gg_array[min_width_mask, 1] = 0
        gg_array[max_width_mask, 1] = max_width
        grasp_group.grasp_group_array = gg_array
        grasp_list, score_list, collision_mask_list = eval_grasp(grasp_group, model_sampled_list, dexmodel_list,
                                                                 transform_pose_list, config, table=table,
                                                                 voxel_size=0.008, TOP_K=TOP_K)
        # remove empty
        grasp_list = [x for x in grasp_list if len(x) != 0]
        score_list = [x for x in score_list if len(x) != 0]
        collision_mask_list = [x for x in collision_mask_list if len(x) != 0]

        if len(grasp_list) == 0:
            grasp_accuracy = np.zeros((TOP_K, len(list_coe_of_friction)))
            scene_accuracy.append(grasp_accuracy)
            grasp_list_list.append([])
            score_list_list.append([])
            collision_list_list.append([])
            # print('\rMean Accuracy for scene:{}'.format(scene_id), np.mean(grasp_accuracy[:, :]),
            #       end='')

        # concat into scene level
        grasp_list, score_list, collision_mask_list = np.concatenate(grasp_list), np.concatenate(
            score_list), np.concatenate(collision_mask_list)
        # sort in scene level
        grasp_confidence = grasp_list[:, 0]
        indices = np.argsort(-grasp_confidence)
        grasp_list, score_list, collision_mask_list = grasp_list[indices], score_list[indices], collision_mask_list[
            indices]

        grasp_accuracy = np.zeros((TOP_K, len(list_coe_of_friction)))
        for fric_idx, fric in enumerate(list_coe_of_friction):
            for k in range(0, TOP_K):
                if k + 1 > len(score_list):
                    grasp_accuracy[k, fric_idx] = np.sum(((score_list <= fric) & (score_list > 0)).astype(int)) / (
                            k + 1)
                else:
                    grasp_accuracy[k, fric_idx] = np.sum(
                        ((score_list[0:k + 1] <= fric) & (score_list[0:k + 1] > 0)).astype(int)) / (k + 1)

        print('\rMean Accuracy for scene:%04d = %.3f' % (
            scene_id, 100.0 * np.mean(grasp_accuracy[:, :])), end='', flush=True)
        scene_accuracy.append(np.mean(grasp_accuracy,axis=0,keepdims=True))
        if not return_list:
            return scene_accuracy
        else:
            return scene_accuracy, grasp_list_list, score_list_list, collision_list_list

# ge = GraspNetEvalComplete(root="/data/graspnet", camera="realsense", split='test')
# path = "logs/dump_csjo"

# res, ap = ge.eval_seen(path, proc=32)
# print("seen")
# print("AP",np.mean(res))
# print(res.shape)
# res = res.transpose(3,0,1,2).reshape(6,-1)
# res = np.mean(res,axis=1)
# print("AP0.4",res[1])
# print("AP0.8",res[3])

# res, ap = ge.eval_similar(path, proc=32)
# print("similar")
# print("AP",np.mean(res))
# print(res.shape)
# res = res.transpose(3,0,1,2).reshape(6,-1)
# res = np.mean(res,axis=1)
# print("AP0.4",res[1])
# print("AP0.8",res[3])

# res, ap = ge.eval_novel(path, proc=32)
# print("novel")
# print(res.shape)
# #delete scene 187 for wrong camera pose
# res = np.delete(res,27,axis=0)
# print("AP",np.mean(res))
# res = res.transpose(3,0,1,2).reshape(6,-1)
# res = np.mean(res,axis=1)
# print("AP0.4",res[1])
# print("AP0.8",res[3])