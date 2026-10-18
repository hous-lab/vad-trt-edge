point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
class_names = [
    'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone',
    'traffic_light', 'pedestrian', 'others'
]
dataset_type = 'B2D_VAD_Dataset'
data_root = 'data/bench2drive'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=True),
    dict(
        type='VADObjectRangeFilter',
        point_cloud_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]),
    dict(
        type='VADObjectNameFilter',
        classes=[
            'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone',
            'traffic_light', 'pedestrian', 'others'
        ]),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.8]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='VADFormatBundle3D',
        class_names=[
            'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone',
            'traffic_light', 'pedestrian', 'others'
        ],
        with_ego=True),
    dict(
        type='CustomCollect3D',
        keys=[
            'gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
            'gt_attr_labels', 'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd',
            'ego_lcf_feat'
        ])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=True),
    dict(
        type='VADObjectRangeFilter',
        point_cloud_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]),
    dict(
        type='VADObjectNameFilter',
        classes=[
            'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone',
            'traffic_light', 'pedestrian', 'others'
        ]),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.8]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='VADFormatBundle3D',
                class_names=[
                    'car', 'van', 'truck', 'bicycle', 'traffic_sign',
                    'traffic_cone', 'traffic_light', 'pedestrian', 'others'
                ],
                with_label=False,
                with_ego=True),
            dict(
                type='CustomCollect3D',
                keys=[
                    'gt_bboxes_3d', 'gt_labels_3d', 'img', 'fut_valid_flag',
                    'ego_his_trajs', 'ego_fut_trajs', 'ego_fut_masks',
                    'ego_fut_cmd', 'ego_lcf_feat', 'gt_attr_labels'
                ])
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=6,
    train=dict(
        type='B2D_VAD_Dataset',
        data_root='data/bench2drive',
        ann_file='data/infos/b2d_infos_train_sub100.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='PhotoMetricDistortionMultiViewImage'),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_attr_label=True),
            dict(
                type='VADObjectRangeFilter',
                point_cloud_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]),
            dict(
                type='VADObjectNameFilter',
                classes=[
                    'car', 'van', 'truck', 'bicycle', 'traffic_sign',
                    'traffic_cone', 'traffic_light', 'pedestrian', 'others'
                ]),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='RandomScaleImageMultiViewImage', scales=[0.8]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='VADFormatBundle3D',
                class_names=[
                    'car', 'van', 'truck', 'bicycle', 'traffic_sign',
                    'traffic_cone', 'traffic_light', 'pedestrian', 'others'
                ],
                with_ego=True),
            dict(
                type='CustomCollect3D',
                keys=[
                    'gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
                    'gt_attr_labels', 'ego_fut_trajs', 'ego_fut_masks',
                    'ego_fut_cmd', 'ego_lcf_feat'
                ])
        ],
        classes=[
            'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone',
            'traffic_light', 'pedestrian', 'others'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=False,
        box_type_3d='LiDAR',
        name_mapping=dict({
            'vehicle.bh.crossbike':
            'bicycle',
            'vehicle.diamondback.century':
            'bicycle',
            'vehicle.gazelle.omafiets':
            'bicycle',
            'vehicle.audi.etron':
            'car',
            'vehicle.chevrolet.impala':
            'car',
            'vehicle.dodge.charger_2020':
            'car',
            'vehicle.dodge.charger_police':
            'car',
            'vehicle.dodge.charger_police_2020':
            'car',
            'vehicle.lincoln.mkz_2017':
            'car',
            'vehicle.lincoln.mkz_2020':
            'car',
            'vehicle.mini.cooper_s_2021':
            'car',
            'vehicle.mercedes.coupe_2020':
            'car',
            'vehicle.ford.mustang':
            'car',
            'vehicle.nissan.patrol_2021':
            'car',
            'vehicle.audi.tt':
            'car',
            'vehicle.ford.crown':
            'car',
            'vehicle.tesla.model3':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked':
            'van',
            'vehicle.ford.ambulance':
            'van',
            'vehicle.carlamotors.firetruck':
            'truck',
            'traffic.speed_limit.30':
            'traffic_sign',
            'traffic.speed_limit.40':
            'traffic_sign',
            'traffic.speed_limit.50':
            'traffic_sign',
            'traffic.speed_limit.60':
            'traffic_sign',
            'traffic.speed_limit.90':
            'traffic_sign',
            'traffic.speed_limit.120':
            'traffic_sign',
            'traffic.stop':
            'traffic_sign',
            'traffic.yield':
            'traffic_sign',
            'traffic.traffic_light':
            'traffic_light',
            'static.prop.warningconstruction':
            'traffic_cone',
            'static.prop.warningaccident':
            'traffic_cone',
            'static.prop.trafficwarning':
            'traffic_cone',
            'static.prop.constructioncone':
            'traffic_cone',
            'walker.pedestrian.0001':
            'pedestrian',
            'walker.pedestrian.0003':
            'pedestrian',
            'walker.pedestrian.0004':
            'pedestrian',
            'walker.pedestrian.0005':
            'pedestrian',
            'walker.pedestrian.0007':
            'pedestrian',
            'walker.pedestrian.0010':
            'pedestrian',
            'walker.pedestrian.0013':
            'pedestrian',
            'walker.pedestrian.0014':
            'pedestrian',
            'walker.pedestrian.0015':
            'pedestrian',
            'walker.pedestrian.0016':
            'pedestrian',
            'walker.pedestrian.0017':
            'pedestrian',
            'walker.pedestrian.0018':
            'pedestrian',
            'walker.pedestrian.0019':
            'pedestrian',
            'walker.pedestrian.0020':
            'pedestrian',
            'walker.pedestrian.0021':
            'pedestrian',
            'walker.pedestrian.0022':
            'pedestrian',
            'walker.pedestrian.0025':
            'pedestrian',
            'walker.pedestrian.0027':
            'pedestrian',
            'walker.pedestrian.0030':
            'pedestrian',
            'walker.pedestrian.0031':
            'pedestrian',
            'walker.pedestrian.0032':
            'pedestrian',
            'walker.pedestrian.0034':
            'pedestrian',
            'walker.pedestrian.0035':
            'pedestrian',
            'walker.pedestrian.0041':
            'pedestrian',
            'walker.pedestrian.0042':
            'pedestrian',
            'walker.pedestrian.0046':
            'pedestrian',
            'walker.pedestrian.0047':
            'pedestrian',
            'static.prop.dirtdebris01':
            'others',
            'static.prop.dirtdebris02':
            'others'
        }),
        map_root='data/bench2drive/maps',
        map_file='data/infos/b2d_map_infos.pkl',
        bev_size=(100, 100),
        queue_length=3,
        past_frames=2,
        future_frames=6,
        point_cloud_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        polyline_points_num=20),
    val=dict(
        type='B2D_VAD_Dataset',
        ann_file='data/infos/b2d_infos_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_attr_label=True),
            dict(
                type='VADObjectRangeFilter',
                point_cloud_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]),
            dict(
                type='VADObjectNameFilter',
                classes=[
                    'car', 'van', 'truck', 'bicycle', 'traffic_sign',
                    'traffic_cone', 'traffic_light', 'pedestrian', 'others'
                ]),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1600, 900),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(type='RandomScaleImageMultiViewImage', scales=[0.8]),
                    dict(type='PadMultiViewImage', size_divisor=32),
                    dict(
                        type='VADFormatBundle3D',
                        class_names=[
                            'car', 'van', 'truck', 'bicycle', 'traffic_sign',
                            'traffic_cone', 'traffic_light', 'pedestrian',
                            'others'
                        ],
                        with_label=False,
                        with_ego=True),
                    dict(
                        type='CustomCollect3D',
                        keys=[
                            'gt_bboxes_3d', 'gt_labels_3d', 'img',
                            'fut_valid_flag', 'ego_his_trajs', 'ego_fut_trajs',
                            'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat',
                            'gt_attr_labels'
                        ])
                ])
        ],
        classes=[
            'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone',
            'traffic_light', 'pedestrian', 'others'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        data_root='data/bench2drive',
        name_mapping=dict({
            'vehicle.bh.crossbike':
            'bicycle',
            'vehicle.diamondback.century':
            'bicycle',
            'vehicle.gazelle.omafiets':
            'bicycle',
            'vehicle.audi.etron':
            'car',
            'vehicle.chevrolet.impala':
            'car',
            'vehicle.dodge.charger_2020':
            'car',
            'vehicle.dodge.charger_police':
            'car',
            'vehicle.dodge.charger_police_2020':
            'car',
            'vehicle.lincoln.mkz_2017':
            'car',
            'vehicle.lincoln.mkz_2020':
            'car',
            'vehicle.mini.cooper_s_2021':
            'car',
            'vehicle.mercedes.coupe_2020':
            'car',
            'vehicle.ford.mustang':
            'car',
            'vehicle.nissan.patrol_2021':
            'car',
            'vehicle.audi.tt':
            'car',
            'vehicle.ford.crown':
            'car',
            'vehicle.tesla.model3':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked':
            'van',
            'vehicle.ford.ambulance':
            'van',
            'vehicle.carlamotors.firetruck':
            'truck',
            'traffic.speed_limit.30':
            'traffic_sign',
            'traffic.speed_limit.40':
            'traffic_sign',
            'traffic.speed_limit.50':
            'traffic_sign',
            'traffic.speed_limit.60':
            'traffic_sign',
            'traffic.speed_limit.90':
            'traffic_sign',
            'traffic.speed_limit.120':
            'traffic_sign',
            'traffic.stop':
            'traffic_sign',
            'traffic.yield':
            'traffic_sign',
            'traffic.traffic_light':
            'traffic_light',
            'static.prop.warningconstruction':
            'traffic_cone',
            'static.prop.warningaccident':
            'traffic_cone',
            'static.prop.trafficwarning':
            'traffic_cone',
            'static.prop.constructioncone':
            'traffic_cone',
            'walker.pedestrian.0001':
            'pedestrian',
            'walker.pedestrian.0003':
            'pedestrian',
            'walker.pedestrian.0004':
            'pedestrian',
            'walker.pedestrian.0005':
            'pedestrian',
            'walker.pedestrian.0007':
            'pedestrian',
            'walker.pedestrian.0010':
            'pedestrian',
            'walker.pedestrian.0013':
            'pedestrian',
            'walker.pedestrian.0014':
            'pedestrian',
            'walker.pedestrian.0015':
            'pedestrian',
            'walker.pedestrian.0016':
            'pedestrian',
            'walker.pedestrian.0017':
            'pedestrian',
            'walker.pedestrian.0018':
            'pedestrian',
            'walker.pedestrian.0019':
            'pedestrian',
            'walker.pedestrian.0020':
            'pedestrian',
            'walker.pedestrian.0021':
            'pedestrian',
            'walker.pedestrian.0022':
            'pedestrian',
            'walker.pedestrian.0025':
            'pedestrian',
            'walker.pedestrian.0027':
            'pedestrian',
            'walker.pedestrian.0030':
            'pedestrian',
            'walker.pedestrian.0031':
            'pedestrian',
            'walker.pedestrian.0032':
            'pedestrian',
            'walker.pedestrian.0034':
            'pedestrian',
            'walker.pedestrian.0035':
            'pedestrian',
            'walker.pedestrian.0041':
            'pedestrian',
            'walker.pedestrian.0042':
            'pedestrian',
            'walker.pedestrian.0046':
            'pedestrian',
            'walker.pedestrian.0047':
            'pedestrian',
            'static.prop.dirtdebris01':
            'others',
            'static.prop.dirtdebris02':
            'others'
        }),
        map_root='data/bench2drive/maps',
        map_file='data/infos/b2d_map_infos.pkl',
        bev_size=(100, 100),
        queue_length=3,
        past_frames=2,
        future_frames=6,
        point_cloud_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        polyline_points_num=20,
        eval_cfg=dict(
            dist_ths=[0.5, 1.0, 2.0, 4.0],
            dist_th_tp=2.0,
            min_recall=0.1,
            min_precision=0.1,
            mean_ap_weight=5,
            class_names=[
                'car', 'van', 'truck', 'bicycle', 'traffic_sign',
                'traffic_cone', 'traffic_light', 'pedestrian'
            ],
            tp_metrics=['trans_err', 'scale_err', 'orient_err', 'vel_err'],
            err_name_maping=dict(
                trans_err='mATE',
                scale_err='mASE',
                orient_err='mAOE',
                vel_err='mAVE',
                attr_err='mAAE'),
            class_range=dict(
                car=(50, 50),
                van=(50, 50),
                truck=(50, 50),
                bicycle=(40, 40),
                traffic_sign=(30, 30),
                traffic_cone=(30, 30),
                traffic_light=(30, 30),
                pedestrian=(40, 40)))),
    test=dict(
        type='B2D_VAD_Dataset',
        data_root='data/bench2drive',
        ann_file='data/infos/b2d_infos_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_attr_label=True),
            dict(
                type='VADObjectRangeFilter',
                point_cloud_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]),
            dict(
                type='VADObjectNameFilter',
                classes=[
                    'car', 'van', 'truck', 'bicycle', 'traffic_sign',
                    'traffic_cone', 'traffic_light', 'pedestrian', 'others'
                ]),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1600, 900),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(type='RandomScaleImageMultiViewImage', scales=[0.8]),
                    dict(type='PadMultiViewImage', size_divisor=32),
                    dict(
                        type='VADFormatBundle3D',
                        class_names=[
                            'car', 'van', 'truck', 'bicycle', 'traffic_sign',
                            'traffic_cone', 'traffic_light', 'pedestrian',
                            'others'
                        ],
                        with_label=False,
                        with_ego=True),
                    dict(
                        type='CustomCollect3D',
                        keys=[
                            'gt_bboxes_3d', 'gt_labels_3d', 'img',
                            'fut_valid_flag', 'ego_his_trajs', 'ego_fut_trajs',
                            'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat',
                            'gt_attr_labels'
                        ])
                ])
        ],
        classes=[
            'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone',
            'traffic_light', 'pedestrian', 'others'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        name_mapping=dict({
            'vehicle.bh.crossbike':
            'bicycle',
            'vehicle.diamondback.century':
            'bicycle',
            'vehicle.gazelle.omafiets':
            'bicycle',
            'vehicle.audi.etron':
            'car',
            'vehicle.chevrolet.impala':
            'car',
            'vehicle.dodge.charger_2020':
            'car',
            'vehicle.dodge.charger_police':
            'car',
            'vehicle.dodge.charger_police_2020':
            'car',
            'vehicle.lincoln.mkz_2017':
            'car',
            'vehicle.lincoln.mkz_2020':
            'car',
            'vehicle.mini.cooper_s_2021':
            'car',
            'vehicle.mercedes.coupe_2020':
            'car',
            'vehicle.ford.mustang':
            'car',
            'vehicle.nissan.patrol_2021':
            'car',
            'vehicle.audi.tt':
            'car',
            'vehicle.ford.crown':
            'car',
            'vehicle.tesla.model3':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked':
            'van',
            'vehicle.ford.ambulance':
            'van',
            'vehicle.carlamotors.firetruck':
            'truck',
            'traffic.speed_limit.30':
            'traffic_sign',
            'traffic.speed_limit.40':
            'traffic_sign',
            'traffic.speed_limit.50':
            'traffic_sign',
            'traffic.speed_limit.60':
            'traffic_sign',
            'traffic.speed_limit.90':
            'traffic_sign',
            'traffic.speed_limit.120':
            'traffic_sign',
            'traffic.stop':
            'traffic_sign',
            'traffic.yield':
            'traffic_sign',
            'traffic.traffic_light':
            'traffic_light',
            'static.prop.warningconstruction':
            'traffic_cone',
            'static.prop.warningaccident':
            'traffic_cone',
            'static.prop.trafficwarning':
            'traffic_cone',
            'static.prop.constructioncone':
            'traffic_cone',
            'walker.pedestrian.0001':
            'pedestrian',
            'walker.pedestrian.0003':
            'pedestrian',
            'walker.pedestrian.0004':
            'pedestrian',
            'walker.pedestrian.0005':
            'pedestrian',
            'walker.pedestrian.0007':
            'pedestrian',
            'walker.pedestrian.0010':
            'pedestrian',
            'walker.pedestrian.0013':
            'pedestrian',
            'walker.pedestrian.0014':
            'pedestrian',
            'walker.pedestrian.0015':
            'pedestrian',
            'walker.pedestrian.0016':
            'pedestrian',
            'walker.pedestrian.0017':
            'pedestrian',
            'walker.pedestrian.0018':
            'pedestrian',
            'walker.pedestrian.0019':
            'pedestrian',
            'walker.pedestrian.0020':
            'pedestrian',
            'walker.pedestrian.0021':
            'pedestrian',
            'walker.pedestrian.0022':
            'pedestrian',
            'walker.pedestrian.0025':
            'pedestrian',
            'walker.pedestrian.0027':
            'pedestrian',
            'walker.pedestrian.0030':
            'pedestrian',
            'walker.pedestrian.0031':
            'pedestrian',
            'walker.pedestrian.0032':
            'pedestrian',
            'walker.pedestrian.0034':
            'pedestrian',
            'walker.pedestrian.0035':
            'pedestrian',
            'walker.pedestrian.0041':
            'pedestrian',
            'walker.pedestrian.0042':
            'pedestrian',
            'walker.pedestrian.0046':
            'pedestrian',
            'walker.pedestrian.0047':
            'pedestrian',
            'static.prop.dirtdebris01':
            'others',
            'static.prop.dirtdebris02':
            'others'
        }),
        map_root='data/bench2drive/maps',
        map_file='data/infos/b2d_map_infos.pkl',
        bev_size=(100, 100),
        queue_length=3,
        past_frames=2,
        future_frames=6,
        point_cloud_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        polyline_points_num=20,
        eval_cfg=dict(
            dist_ths=[0.5, 1.0, 2.0, 4.0],
            dist_th_tp=2.0,
            min_recall=0.1,
            min_precision=0.1,
            mean_ap_weight=5,
            class_names=[
                'car', 'van', 'truck', 'bicycle', 'traffic_sign',
                'traffic_cone', 'traffic_light', 'pedestrian'
            ],
            tp_metrics=['trans_err', 'scale_err', 'orient_err', 'vel_err'],
            err_name_maping=dict(
                trans_err='mATE',
                scale_err='mASE',
                orient_err='mAOE',
                vel_err='mAVE',
                attr_err='mAAE'),
            class_range=dict(
                car=(50, 50),
                van=(50, 50),
                truck=(50, 50),
                bicycle=(40, 40),
                traffic_sign=(30, 30),
                traffic_cone=(30, 30),
                traffic_light=(30, 30),
                pedestrian=(40, 40)))),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))
evaluation = dict(
    interval=999, pipeline=None, metric='bbox', map_metric='chamfer')
checkpoint_config = dict(interval=1, max_keep_ckpts=2)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/vad_small_b2d_fast'
load_from = None
resume_from = None
workflow = [('train', 1)]
voxel_size = [0.15, 0.15, 4]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
NameMapping = dict({
    'vehicle.bh.crossbike':
    'bicycle',
    'vehicle.diamondback.century':
    'bicycle',
    'vehicle.gazelle.omafiets':
    'bicycle',
    'vehicle.audi.etron':
    'car',
    'vehicle.chevrolet.impala':
    'car',
    'vehicle.dodge.charger_2020':
    'car',
    'vehicle.dodge.charger_police':
    'car',
    'vehicle.dodge.charger_police_2020':
    'car',
    'vehicle.lincoln.mkz_2017':
    'car',
    'vehicle.lincoln.mkz_2020':
    'car',
    'vehicle.mini.cooper_s_2021':
    'car',
    'vehicle.mercedes.coupe_2020':
    'car',
    'vehicle.ford.mustang':
    'car',
    'vehicle.nissan.patrol_2021':
    'car',
    'vehicle.audi.tt':
    'car',
    'vehicle.ford.crown':
    'car',
    'vehicle.tesla.model3':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked':
    'van',
    'vehicle.ford.ambulance':
    'van',
    'vehicle.carlamotors.firetruck':
    'truck',
    'traffic.speed_limit.30':
    'traffic_sign',
    'traffic.speed_limit.40':
    'traffic_sign',
    'traffic.speed_limit.50':
    'traffic_sign',
    'traffic.speed_limit.60':
    'traffic_sign',
    'traffic.speed_limit.90':
    'traffic_sign',
    'traffic.speed_limit.120':
    'traffic_sign',
    'traffic.stop':
    'traffic_sign',
    'traffic.yield':
    'traffic_sign',
    'traffic.traffic_light':
    'traffic_light',
    'static.prop.warningconstruction':
    'traffic_cone',
    'static.prop.warningaccident':
    'traffic_cone',
    'static.prop.trafficwarning':
    'traffic_cone',
    'static.prop.constructioncone':
    'traffic_cone',
    'walker.pedestrian.0001':
    'pedestrian',
    'walker.pedestrian.0003':
    'pedestrian',
    'walker.pedestrian.0004':
    'pedestrian',
    'walker.pedestrian.0005':
    'pedestrian',
    'walker.pedestrian.0007':
    'pedestrian',
    'walker.pedestrian.0010':
    'pedestrian',
    'walker.pedestrian.0013':
    'pedestrian',
    'walker.pedestrian.0014':
    'pedestrian',
    'walker.pedestrian.0015':
    'pedestrian',
    'walker.pedestrian.0016':
    'pedestrian',
    'walker.pedestrian.0017':
    'pedestrian',
    'walker.pedestrian.0018':
    'pedestrian',
    'walker.pedestrian.0019':
    'pedestrian',
    'walker.pedestrian.0020':
    'pedestrian',
    'walker.pedestrian.0021':
    'pedestrian',
    'walker.pedestrian.0022':
    'pedestrian',
    'walker.pedestrian.0025':
    'pedestrian',
    'walker.pedestrian.0027':
    'pedestrian',
    'walker.pedestrian.0030':
    'pedestrian',
    'walker.pedestrian.0031':
    'pedestrian',
    'walker.pedestrian.0032':
    'pedestrian',
    'walker.pedestrian.0034':
    'pedestrian',
    'walker.pedestrian.0035':
    'pedestrian',
    'walker.pedestrian.0041':
    'pedestrian',
    'walker.pedestrian.0042':
    'pedestrian',
    'walker.pedestrian.0046':
    'pedestrian',
    'walker.pedestrian.0047':
    'pedestrian',
    'static.prop.dirtdebris01':
    'others',
    'static.prop.dirtdebris02':
    'others'
})
eval_cfg = dict(
    dist_ths=[0.5, 1.0, 2.0, 4.0],
    dist_th_tp=2.0,
    min_recall=0.1,
    min_precision=0.1,
    mean_ap_weight=5,
    class_names=[
        'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone',
        'traffic_light', 'pedestrian'
    ],
    tp_metrics=['trans_err', 'scale_err', 'orient_err', 'vel_err'],
    err_name_maping=dict(
        trans_err='mATE',
        scale_err='mASE',
        orient_err='mAOE',
        vel_err='mAVE',
        attr_err='mAAE'),
    class_range=dict(
        car=(50, 50),
        van=(50, 50),
        truck=(50, 50),
        bicycle=(40, 40),
        traffic_sign=(30, 30),
        traffic_cone=(30, 30),
        traffic_light=(30, 30),
        pedestrian=(40, 40)))
num_classes = 9
map_classes = [
    'Broken', 'Solid', 'SolidSolid', 'Center', 'TrafficLight', 'StopSign'
]
map_num_vec = 100
map_fixed_ptsnum_per_gt_line = 20
map_fixed_ptsnum_per_pred_line = 20
map_eval_use_same_gt_sample_num_flag = True
map_num_classes = 6
past_frames = 2
future_frames = 6
_dim_ = 256
_pos_dim_ = 128
_ffn_dim_ = 512
_num_levels_ = 1
bev_h_ = 100
bev_w_ = 100
queue_length = 3
total_epochs = 2
model = dict(
    type='VAD',
    use_grid_mask=True,
    video_test_mode=True,
    pretrained=dict(img='ckpts/resnet50-19c8e357.pth'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=1,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='VADHead',
        map_thresh=0.5,
        dis_thresh=0.2,
        pe_normalization=True,
        tot_epoch=12,
        use_traj_lr_warmup=False,
        query_thresh=0.0,
        query_use_fix_pad=False,
        ego_his_encoder=None,
        ego_lcf_feat_idx=None,
        valid_fut_ts=6,
        ego_fut_mode=6,
        ego_agent_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.0)
                ],
                feedforward_channels=512,
                ffn_dropout=0.0,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        ego_map_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.0)
                ],
                feedforward_channels=512,
                ffn_dropout=0.0,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        motion_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.0)
                ],
                feedforward_channels=512,
                ffn_dropout=0.0,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        motion_map_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.0)
                ],
                feedforward_channels=512,
                ffn_dropout=0.0,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        use_pe=True,
        bev_h=100,
        bev_w=100,
        num_query=300,
        num_classes=9,
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        map_num_vec=100,
        map_num_classes=6,
        map_num_pts_per_vec=20,
        map_num_pts_per_gt_vec=20,
        map_query_embed_type='instance_pts',
        map_transform_method='minmax',
        map_gt_shift_pts_pattern='v2',
        map_dir_interval=1,
        map_code_size=2,
        map_code_weights=[1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='VADPerceptionTransformer',
            map_num_vec=100,
            map_num_pts_per_vec=20,
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=256,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=3,
                pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=256,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=256,
                                num_points=8,
                                num_levels=1),
                            embed_dims=256)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=3,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.0),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=256,
                            num_levels=1)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            map_decoder=dict(
                type='MapDetectionTransformerDecoder',
                num_layers=3,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.0),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=256,
                            num_levels=1)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='CustomNMSFreeCoder',
            post_center_range=[-20, -35, -10.0, 20, 35, 10.0],
            pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
            max_num=100,
            voxel_size=[0.15, 0.15, 4],
            num_classes=9),
        map_bbox_coder=dict(
            type='MapNMSFreeCoder',
            post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
            pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
            max_num=50,
            voxel_size=[0.15, 0.15, 4],
            num_classes=6),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=128,
            row_num_embed=100,
            col_num_embed=100),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_traj=dict(type='L1Loss', loss_weight=0.2),
        loss_traj_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.2),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_map_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_map_bbox=dict(type='L1Loss', loss_weight=0.0),
        loss_map_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_map_pts=dict(type='PtsL1Loss', loss_weight=1.0),
        loss_map_dir=dict(type='PtsDirCosLoss', loss_weight=0.005),
        loss_plan_reg=dict(type='L1Loss', loss_weight=1.0),
        loss_plan_bound=dict(
            type='PlanMapBoundLoss', loss_weight=1.0, dis_thresh=1.0),
        loss_plan_col=dict(type='PlanCollisionLoss', loss_weight=1.0),
        loss_plan_dir=dict(type='PlanMapDirectionLoss', loss_weight=0.5)),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=[0.15, 0.15, 4],
            point_cloud_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]),
            map_assigner=dict(
                type='MapHungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(
                    type='BBoxL1Cost', weight=0.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
                pts_cost=dict(type='OrderedPtsL1Cost', weight=1.0),
                pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]))))
info_root = 'data/infos'
map_root = 'data/bench2drive/maps'
map_file = 'data/infos/b2d_map_infos.pkl'
ann_file_train = 'data/infos/b2d_infos_train.pkl'
ann_file_val = 'data/infos/b2d_infos_val.pkl'
ann_file_test = 'data/infos/b2d_infos_val.pkl'
inference_only_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.8]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='VADFormatBundle3D',
                class_names=[
                    'car', 'van', 'truck', 'bicycle', 'traffic_sign',
                    'traffic_cone', 'traffic_light', 'pedestrian', 'others'
                ],
                with_label=False,
                with_ego=True),
            dict(type='CustomCollect3D', keys=['img', 'ego_fut_cmd'])
        ])
]
optimizer = dict(
    type='AdamW',
    lr=0.0002,
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    by_epoch=False,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=20,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.01)
runner = dict(type='EpochBasedRunner', max_epochs=2)
custom_hooks = [dict(type='CustomSetEpochInfoHook')]
