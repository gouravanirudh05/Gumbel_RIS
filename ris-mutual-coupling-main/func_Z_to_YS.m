function [yRT,yRI,yIT,sRT,sRI,sIT] = func_Z_to_YS(zRT,zRI,zIT,ZII)

Z0 = 50;
NI = size(ZII,1);

yRI = - (zRI / ZII) / Z0;
yIT = - (ZII \ zIT) / Z0;
yRT = (- zRT + zRI / ZII * zIT) / Z0^2;

sRI = zRI / (ZII + Z0 * eye(NI));
sIT = (ZII + Z0 * eye(NI)) \ zIT;
sRT = (zRT - zRI / (ZII + Z0 * eye(NI)) * zIT) / (2*Z0);

end