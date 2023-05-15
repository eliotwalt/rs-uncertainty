# earthengine authenticate --auth_mode=notebook
mapfile -t taskList < <( earthengine task list | cut -d " " -f1)
for taskId in ${taskList[@]}
do    
    earthengine task cancel ${taskId}
done

