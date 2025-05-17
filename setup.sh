echo "-- Downloading the KuaiRec dataset"
wget -q --show-progress --no-check-certificate 'https://drive.usercontent.google.com/download?id=1qe5hOSBxzIuxBb1G_Ih5X-O65QElollE&export=download&confirm=t&uuid=b2002093-cc6e-4bd5-be47-9603f0b33470
' -O KuaiRec.zip

echo "-- Inflating"
unzip KuaiRec.zip -d KuaiRec

echo "-- Moving data"
mv "KuaiRec/KuaiRec 2.0/data" .

echo "-- Cleanup"
rm -rf KuaiRec*
