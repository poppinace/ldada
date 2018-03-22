function opt = paramInit

opt.rootdir = ''; % revise if needed

opt.dataset = 'Office-Caltech10';
% opt.dataset = 'Office';
% opt.dataset = 'SatelliteScene5';
% opt.dataset = 'MTFS3';

opt.modelType = 'VGG-M';
% opt.modelType = 'VGG-VD-16';

opt.featureType = 'FC';

% set source domain and target domain
opt.sourcedir = 'amazon';
opt.targetdir = 'webcam';

opt.nclasstrain = -1; % -1 for full protocol, 20 for sampling protocol
opt.ntrials = 20;

opt.center = true;

% path seting
opt.datasetdir = fullfile(opt.rootdir, 'data', opt.dataset); 

% set annotation path
opt.annotationdir = fullfile(opt.datasetdir, 'annotations');

opt.imagedir = 'images';

opt.cachedir = 'cache';
opt.modeldir = 'model';

switch opt.dataset
  case 'Office-Caltech10'
    opt.classes={...
        'back_pack'
        'bike'
        'calculator'
        'headphones'
        'keyboard'
        'laptop_computer'
        'monitor'
        'mouse'
        'mug'
        'projector'
    };
  case 'Office'
    opt.classes={...
        'back_pack'
        'bike'
        'bike_helmet'
        'bookcase'
        'bottle'
        'calculator'
        'desk_chair'
        'desk_lamp'
        'desktop_computer'
        'file_cabinet'
        'headphones'
        'keyboard'
        'laptop_computer'
        'letter_tray'
        'mobile_phone'
        'monitor'
        'mouse'
        'mug'
        'paper_notebook'
        'pen'
        'phone'
        'printer'
        'projector'
        'punchers'
        'ring_binder'
        'ruler'
        'scissors'
        'speaker'
        'stapler'
        'tape_dispenser'
        'trash_can'
    };
  case 'SatelliteScene5'
    opt.classes={...
        'field'
        'forest'
        'industry'
        'residential'
        'river'
    };
  case 'MTFS3'
    opt.classes={...
        'non-flowering'
        'partially-flowering'
        'fully-flowering'
    };
  otherwise
    error('Unsupported dataset')
end

% parameter setting
opt.nclasses = length(opt.classes);

% ldada setting
opt.ldada.maxiter = 10;
opt.ldada.predictor = 'ldada'; % 'ldada' or 'svm'
opt.ldada.verbose = true;

end