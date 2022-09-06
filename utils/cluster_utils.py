def start_cluster(profile = None, mode=None, backend=None, cores=16, processes=8, memory='16GB', total_cores=32, port=12123):
    from time import sleep
    if backend is None:
        backend='dask'
    if mode is None:
        mode = 'cluster'
    if profile is None:
        profile = 'local'
    if profile == "local":
        import dask
        from dask.distributed import Client, LocalCluster
        from multiprocessing.pool import ThreadPool
        if mode == 'client':
            dask.config.set(pool=ThreadPool(8))
        elif mode == 'cluster':
            dask.config.set(pool=ThreadPool(8))
        elif mode == 'auton':
#             dask.config.set(pool=ThreadPool(processes))
            pass
        else:
            raise NotImplementedError
            
        from MyModel import MyModel
        if mode == 'client' or mode == 'cluster':
            client = Client(processes=4)
        elif mode == 'auton':
            cluster = LocalCluster(n_workers=8, threads_per_worker=6, processes=True, memory_limit='200GB', dashboard_address=':8787')
            client = Client(cluster)
        else:
            raise NotImplementedError
        backend = 'dask'
        filename = "fake_reviews_result-job-{0}.txt"

        return client, None
    elif profile == 'SSH':
        host_list = ['lov1', 'lov2']#, 'lov3'#], 'lov4']#, 'lov5', 'lov6', 'low1', 'ari']
        from dask.distributed import Client, SSHCluster
#         cluster = SSHCluster(
#             host_list,
#             connect_options={"known_hosts": None, 'username':'nshajari', "client_keys":'/zfsauton2/home/nshajari/.ssh/id_rsa'},
#             scheduler_options={"port": 0, "dashboard_address": ":8787"},
#             worker_options={'nthreads':4})
        cluster = SSHCluster(
            hosts=host_list[0],
#             scheduler_port=0,
#             worker_addrs=host_list,
            connect_options={"known_hosts": None, 'username':'nshajari', "client_keys":'/zfsauton2/home/nshajari/.ssh/id_rsa'},
            scheduler_options={"port": 0, "dashboard_address": ":8787"},
            worker_options={'nthreads':4})

#             worker_module='dask_cuda.dask_cuda_worker')
#         cluster.wait()
        client = Client(cluster, timeout='120s')
        return client, cluster
    elif profile == 'dask':
        print ("salam")
        from dask_jobqueue import SLURMCluster
        cluster = SLURMCluster(
                               cores=cores,
                               processes=processes, 
        #                      job_mem='100GB',
                             memory=memory,
    #                          project='test',
                             nanny=True,
                            interface='ib0',
                            dashboard_address=':'+str(port),
                            silence_log='debug',
                            walltime='6:30:00',
                            log_directory='/home/nshajari/master_thesis/dask_logs')
#                                 walltime='00:03:00')
        cluster.scale(cores=total_cores)  # Start 100 workers in 100 jobs that match the description above
        from dask.distributed import Client
    #     sleep(10)
        client = Client(cluster, processes=False, timeout='120s')

        return client, cluster
    else:
        raise NotImplementedError
if __name__ == '__main__':
    pass
def test():
    print ("salam")
