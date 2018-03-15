function log4m_make_instance(log_file)
% LOG4M_MAKE_INSTANCE  Make instance of log4m
% 
% (c) Tomoya Sakai, The University of Tokyo, Japan.
%     sakai@ms.k.u-tokyo.ac.jp
global LOG;

log_dir = fullfile(pwd, 'log');
if ~exist(log_dir, 'dir')
    mkdir(log_dir);
end
LOG = log4m.getLogger(fullfile(log_dir, log_file));

end
