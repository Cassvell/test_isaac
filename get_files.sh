#FTP
server=lisn.igp.gob.pe
port=21
#type: FTP
#Folder: /
U=issacc
P=AtKiTbXiUNqfTgSjHdG6

echo 'descargar datos magneticos'

#find $fdir -type f -name "*QRO.csv"
read -p "enter station code (eg. huan, piur, nazc): " st

read -p "Enter the initial date with the format (yyyymmdd): " idate
read -p "Enter the final date with the format (yyyymmdd): " fdate

#echo $month_dir
#echo $date_name

local_path="$HOME/MEGAsync/datos/jicamarca/${st}"
if [[ -e ${local_path} ]]
then
	echo "$directory {local_path} exist "	
else	
	mkdir ${local_path}
	echo "$directory {local_path} created!!"
fi	
idate=$(date -d "${idate}" +"%Y-%m-%d")
fdate=$(date -d "${fdate}" +"%Y-%m-%d")

while [[ "$idate" < "$fdate" ]];do	

	year_dir=$(date -d "$idate" +"%Y")
	month_dir=$(date -d "$idate" +"%m")
	day=$(date -d "$idate" +"%d")
        yy=${year_dir:2:2}

	file="${local_path}/${st}_${yy}${month_dir}${day}.min.gz"
	
	if [ -f "$file" ]; then	
		echo "$file exist"    
	else
		#echo "$file does not exist"	
                server_path="/IGP/${st}/$year_dir/$month_dir/minute"
                wget --ftp-user=$U --ftp-password=$P @ ftp://${server}${server_path}/${st}_${yy}${month_dir}${day}.min.gz -P $local_path
	       
	fi	
	idate=$(date -I -d "$idate + 1 day")
	#echo ${server_path}/huan_${date_name}.min.gz
		#echo $DOY
done

for g in ${local_path}/*.gz; do gunzip $g; done
