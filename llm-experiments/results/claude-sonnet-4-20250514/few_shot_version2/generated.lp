```asp
square(1..8,1..8).

% Generate all possible moves for the side to move
possible_move(Color,X1,Y1,X2,Y2) :- 
    side_to_move(Color),
    piece(Color,pawn,X1,Y1), Color=white,
    Y2=Y1+1, X2=X1, square(X2,Y2),
    not piece(_,_,X2,Y2).

possible_move(Color,X1,Y1,X2,Y2) :- 
    side_to_move(Color),
    piece(Color,pawn,X1,Y1), Color=white, Y1=2,
    Y2=Y1+2, X2=X1, square(X2,Y2),
    not piece(_,_,X2,Y2).

possible_move(Color,X1,Y1,X2,Y2) :- 
    side_to_move(Color),
    piece(Color,pawn,X1,Y1), Color=white,
    Y2=Y1+1, (X2=X1+1; X2=X1-1), square(X2,Y2),
    piece(OppColor,_,X2,Y2), Color!=OppColor.

possible_move(Color,X1,Y1,X2,Y2) :- 
    side_to_move(Color),
    piece(Color,pawn,X1,Y1), Color=black,
    Y2=Y1-1, X2=X1, square(X2,Y2),
    not piece(_,_,X2,Y2).

possible_move(Color,X1,Y1,X2,Y2) :- 
    side_to_move(Color),
    piece(Color,pawn,X1,Y1), Color=black, Y1=7,
    Y2=Y1-2, X2=X1, square(X2,Y2),
    not piece(_,_,X2,Y2).

possible_move(Color,X1,Y1,X2,Y2) :- 
    side_to_move(Color),
    piece(Color,pawn,X1,Y1), Color=black,
    Y2=Y1-1, (X2=X1+1; X2=X1-1), square(X2,Y2),
    piece(OppColor,_,X2,Y2), Color!=OppColor.

possible_move(Color,X1,Y1,X2,Y2) :- 
    side_to_move(Color),
    piece(Color,rook,X1,Y1),
    square(X2,Y2), (X2=X1; Y2=Y1), (X2,Y2)!=(X1,Y1),
    not piece(Color,_,X2,Y2),
    not blocked_path(X1,Y1,X2,Y2).

possible_move(Color,X1,Y1,X2,Y2) :- 
    side_to_move(Color),
    piece(Color,bishop,X1,Y1),
    square(X2,Y2), |X2-X1|=|Y2-Y1|, (X2,Y2)!=(X1,Y1),
    not piece(Color,_,X2,Y2),
    not blocked_diagonal(X1,Y1,X2,Y2).

possible_move(Color,X1,Y1,X2,Y2) :- 
    side_to_move(Color),
    piece(Color,queen,X1,Y1),
    square(X2,Y2), ((X2=X1; Y2=Y1); |X2-X1|=|Y2-Y1|), (X2,Y2)!=(X1,Y1),
    not piece(Color,_,X2,Y2),
    not blocked_path(X1,Y1,X2,Y2),
    not blocked_diagonal(X1,Y1,X2,Y2).

possible_move(Color,X1,Y1,X2,Y2) :- 
    side_to_move(Color),
    piece(Color,knight,X1,Y1),
    square(X2,Y2),
    ((X2=X1+2,Y2=Y1+1); (X2=X1+2,Y2=Y1-1); (X2=X1-2,Y2=Y1+1); (X2=X1-2,Y2=Y1-1);
     (X2=X1+1,Y2=Y1+2); (X2=X1+1,Y2=Y1-2); (X2=X1-1,Y2=Y1+2); (X2=X1-1,Y2=Y1-2)),
    not piece(Color,_,X2,Y2).

possible_move(Color,X1,Y1,X2,Y2) :- 
    side_to_move(Color),
    piece(Color,king,X1,Y1),
    square(X2,Y2), |X2-X1|<=1, |Y2-Y1|<=1, (X2,Y2)!=(X1,Y1),
    not piece(Color,_,X2,Y2).

blocked_path(X1,Y1,X2,Y2) :- 
    X1=X2, Y1<Y2,
    piece(_,_,X1,Y), Y1<Y<Y2.

blocked_path(X1,Y1,X2,Y2) :- 
    X1=X2, Y1>Y2,
    piece(_,_,X1,Y), Y2<Y<Y1.

blocked_path(X1,Y1,X2,Y2) :- 
    Y1=Y2, X1<X2,
    piece(_,_,X,Y1), X1<X<X2.

blocked_path(X1,Y1,X2,Y2) :- 
    Y1=Y2, X1>X2,
    piece(_,_,X,Y1), X2<X<X1.

blocked_diagonal(X1,Y1,X2,Y2) :- 
    X1<X2, Y1<Y2,
    piece(_,_,X,Y), X1<X<X2, Y1<Y<Y2, X-X1=Y-Y1.

blocked_diagonal(X1,Y1,X2,Y2) :- 
    X1<X2, Y1>Y2,
    piece(_,_,X,Y), X1<X<X2, Y2<Y<Y1, X-X1=Y1-Y.

blocked_diagonal(X1,Y1,X2,Y2) :- 
    X1>X2, Y1<Y2,
    piece(_,_,X,Y), X2<X<X1, Y1<Y<Y2, X1-X=Y-Y1.

blocked_diagonal(X1,Y1,X2,Y2) :- 
    X1>X2, Y1>Y2,
    piece(_,_,X,Y), X2<X<X1, Y2<Y<Y1, X1-X=Y1-Y.

% Apply move to create new position
new_piece(Color,Type,X2,Y2,X1,Y1,X2,Y2) :- 
    possible_move(Color,X1,Y1,X2,Y2),
    piece(Color,Type,X1,Y1).

new_piece(Color,Type,X,Y,X1,Y1,X2,Y2) :- 
    possible_move(_,X1,Y1,X2,Y2),
    piece(Color,Type,X,Y),
    (X,Y)!=(X1,Y1), (X,Y)!=(X2,Y2).

% Detect attacks in new position
attack_square(X,Y,X1,Y1,X2,Y2) :- 
    new_piece(Color,pawn,PX,PY,X1,Y1,X2,Y2), Color=white,
    X=PX+1, Y=PY+1, square(X,Y).

attack_square(X,Y,X1,Y1,X2,Y2) :- 
    new_piece(Color,pawn,PX,PY,X1,Y1,X2,Y2), Color=white,
    X=PX-1, Y=PY+1, square(X,Y).

attack_square(X,Y,X1,Y1,X2,Y2) :- 
    new_piece(Color,pawn,PX,PY,X1,Y1,X2,Y2), Color=black,
    X=PX+1, Y=PY-1, square(X,Y).

attack_square(X,Y,X1,Y1,X2,Y2) :- 
    new_piece(Color,pawn,PX,PY,X1,Y1,X2,Y2), Color=black,
    X=PX-1, Y=PY-1, square(X,Y).

attack_square(X,Y,X1,Y1,X2,Y2) :- 
    new_piece(Color,rook,RX,RY,X1,Y1,X2,Y2),
    square(X,Y), (X=RX; Y=RY), (X,Y)!=(RX,RY),
    not new_blocked_path(RX,RY,X,Y,X1,Y1,X2,Y2).

attack_square(X,Y,X1,Y1,X2,Y2) :- 
    new_piece(Color,bishop,BX,BY,X1,Y1,X2,Y2),
    square(X,Y), |X-BX|=|Y-BY|, (X,Y)!=(BX,BY),
    not new_blocked_diagonal(BX,BY,X,Y,X1,Y1,X2,Y2).

attack_square(X,Y,X1,Y1,X2,Y2) :- 
    new_piece(Color,queen,QX,QY,X1,Y1,X2,Y2),
    square(X,Y), ((X=QX; Y=QY); |X-QX|=|Y-QY|), (X,Y)!=(QX,QY),
    not new_blocked_path(QX,QY,X,Y,X1,Y1,X2,Y2),
    not new_blocked_diagonal(QX,QY,X,Y,X1,Y1,X2,Y2).

attack_square(X,Y,X1,Y1,X2,Y2) :- 
    new_piece(Color,knight,NX,NY,X1,Y1,X2,Y2),
    square(X,Y),
    ((X=NX+2,Y=NY+1); (X=NX+2,Y=NY-1); (X=NX-2,Y=NY+1); (X=NX-2,Y=NY-1);
     (X=NX+1,Y=NY+2); (X=NX+1,Y=NY-2); (X=NX-1,Y=NY+2); (X=NX-1,Y=NY-2)).

attack_square(X,Y,X1,Y1,X2,Y2) :- 
    new_piece(Color,king,KX,KY,X1,Y1,X2,Y2),
    square(X,Y), |X-KX|<=1, |Y-KY|<=1, (X,Y)!=(KX,KY).

new_blocked_path(PX1,PY1,PX2,PY2,X1,Y1,X2,Y2) :- 
    PX1=PX2, PY1<PY2,
    new_piece(_,_,PX1,PY,X1,Y1,X2,Y2), PY1<PY<PY2.

new_blocked_path(PX1,PY1,PX2,PY2,X1,Y1,X2,Y2) :- 
    PX1=PX2, PY1>PY2,
    new_piece(_,_,PX1,PY,X1,Y1,X2,Y2), PY2<PY<PY1.

new_blocked_path(PX1,PY1,PX2,PY2,X1,Y1,X2,Y2) :- 
    PY1=PY2, PX1<PX2,
    new_piece(_,_,PX,PY1,X1,Y1,X2,Y2), PX1<PX<PX2.

new_blocked_path(PX1,PY1,PX2,PY2,X1,Y1,X2,Y2) :- 
    PY1=PY2, PX1>PX2,
    new_piece(_,_,PX,PY1,X1,Y1,X2,Y2), PX2<PX<PX1.

new_blocked_diagonal(PX1,PY1,PX2,PY2,X1,Y1,X2,Y2) :- 
    PX1<PX2, PY1<PY2,
    new_piece(_,_,PX,PY,X1,Y1,X2,Y2), PX1<PX<PX2, PY1<PY<PY2, PX-PX1=PY-PY1.

new_blocked_diagonal(PX1,PY1,PX2,PY2,X1,Y1,X2,Y2) :- 
    PX1<PX2, PY1>PY2,
    new_piece(_,_,PX,PY,X1,Y1,X2,Y2), PX1<PX<PX2, PY2<PY<PY1, PX-PX1=PY1-PY.

new_blocked_diagonal(PX1,PY1,PX2,PY2,X1,Y1,X2,Y2) :- 
    PX1>PX2, PY1<PY2,
    new_piece(_,_,PX,PY,X1,Y1,X2,Y2), PX2<PX<PX1, PY1<PY<PY2, PX1-PX=PY-PY1.

new_blocked_diagonal(PX1,PY1,PX2,PY2,X1,Y1,X2,Y2) :- 
    PX1>PX2, PY1>PY2,
    new_piece(_,_,PX,PY,X1,Y1,X2,Y2), PX2<PX<PX1, PY2<PY<PY1, PX1-PX=PY1-PY.

% Detect check on opponent king
opponent_color(white) :- side_to_move(black).
opponent_color(black) :- side_to_move(white).

opponent_in_check(X1,Y1,X2,Y2) :- 
    possible_move(_,X1,Y1,X2,Y2),
    opponent_color(OppColor),
    king(OppColor,KX,KY),
    side_to_move(Color),
    attack_square(KX,KY,X1,Y1,X2,Y2),
    new_piece(Color,_,_,_,X1,Y1,X2,Y2).

% Generate opponent escape moves
opponent_move(OppColor,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2) :- 
    opponent_in_check(X1,Y1,X2,Y2),
    opponent_color(OppColor),
    new_piece(OppColor,pawn,OX1,OY1,X1,Y1,X2,Y2), OppColor=white,
    OY2=OY1+1, OX2=OX1, square(OX2,OY2),
    not new_piece(_,_,OX2,OY2,X1,Y1,X2,Y2).

opponent_move(OppColor,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2) :- 
    opponent_in_check(X1,Y1,X2,Y2),
    opponent_color(OppColor),
    new_piece(OppColor,pawn,OX1,OY1,X1,Y1,X2,Y2), OppColor=white, OY1=2,
    OY2=OY1+2, OX2=OX1, square(OX2,OY2),
    not new_piece(_,_,OX2,OY2,X1,Y1,X2,Y2).

opponent_move(OppColor,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2) :- 
    opponent_in_check(X1,Y1,X2,Y2),
    opponent_color(OppColor),
    new_piece(OppColor,pawn,OX1,OY1,X1,Y1,X2,Y2), OppColor=white,
    OY2=OY1+1, (OX2=OX1+1; OX2=OX1-1), square(OX2,OY2),
    new_piece(Color,_,OX2,OY2,X1,Y1,X2,Y2), Color!=OppColor.

opponent_move(OppColor,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2) :- 
    opponent_in_check(X1,Y1,X2,Y2),
    opponent_color(OppColor),
    new_piece(OppColor,pawn,OX1,OY1,X1,Y1,X2,Y2), OppColor=black,
    OY2=OY1-1, OX2=OX1, square(OX2,OY2),
    not new_piece(_,_,OX2,OY2,X1,Y1,X2,Y2).

opponent_move(OppColor,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2) :- 
    opponent_in_check(X1,Y1,X2,Y2),
    opponent_color(OppColor),
    new_piece(OppColor,pawn,OX1,OY1,X1,Y1,X2,Y2), OppColor=black, OY1=7,
    OY2=OY1-2, OX2=OX1, square(OX2,OY2),
    not new_piece(_,_,OX2,OY2,X1,Y1,X2,Y2).

opponent_move(OppColor,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2) :- 
    opponent_in_check(X1,Y1,X2,Y2),
    opponent_color(OppColor),
    new_piece(OppColor,pawn,OX1,OY1,X1,Y1,X2,Y2), OppColor=black,
    OY2=OY1-1, (OX2=OX1+1; OX2=OX1-1), square(OX2,OY2),
    new_piece(Color,_,OX2,OY2,X1,Y1,X2,Y2), Color!=OppColor.

opponent_move(OppColor,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2) :- 
    opponent_in_check(X1,Y1,X2,Y2),
    opponent_color(OppColor),
    new_piece(OppColor,rook,OX1,OY1,X1,Y1,X2,Y2),
    square(OX2,OY2), (OX2=OX1; OY2=OY1), (OX2,OY2)!=(OX1,OY1),
    not new_piece(OppColor,_,OX2,OY2,X1,Y1,X2,Y2),
    not new_blocked_path(OX1,OY1,OX2,OY2,X1,Y1,X2,Y2).

opponent_move(OppColor,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2) :- 
    opponent_in_check(X1,Y1,X2,Y2),
    opponent_color(OppColor),
    new_piece(OppColor,bishop,OX1,OY1,X1,Y1,X2,Y2),
    square(OX2,OY2), |OX2-OX1|=|OY2-OY1|, (OX2,OY2)!=(OX1,OY1),
    not new_piece(OppColor,_,OX2,OY2,X1,Y1,X2,Y2),
    not new_blocked_diagonal(OX1,OY1,OX2,OY2,X1,Y1,X2,Y2).

opponent_move(OppColor,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2) :- 
    opponent_in_check(X1,Y1,X2,Y2),
    opponent_color(OppColor),
    new_piece(OppColor,queen,OX1,OY1,X1,Y1,X2,Y2),
    square(OX2,OY2), ((OX2=OX1; OY2=OY1); |OX2-OX1|=|OY2-OY1|), (OX2,OY2)!=(OX1,OY1),
    not new_piece(OppColor,_,OX2,OY2,X1,Y1,X2,Y2),
    not new_blocked_path(OX1,OY1,OX2,OY2,X1,Y1,X2,Y2),
    not new_blocked_diagonal(OX1,OY1,OX2,OY2,X1,Y1,X2,Y2).

opponent_move(OppColor,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2) :- 
    opponent_in_check(X1,Y1,X2,Y2),
    opponent_color(OppColor),
    new_piece(OppColor,knight,OX1,OY1,X1,Y1,X2,Y2),
    square(OX2,OY2),
    ((OX2=OX1+2,OY2=OY1+1); (OX2=OX1+2,OY2=OY1-1); (OX2=OX1-2,OY2=OY1+1); (OX2=OX1-2,OY2=OY1-1);
     (OX2=OX1+1,OY2=OY1+2); (OX2=OX1+1,OY2=OY1-2); (OX2=OX1-1,OY2=OY1+2); (OX2=OX1-1,OY2=OY1-2)),
    not new_piece(OppColor,_,OX2,OY2,X1,Y1,X2,Y2).

opponent_move(OppColor,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2) :- 
    opponent_in_check(X1,Y1,X2,Y2),
    opponent_color(OppColor),
    new_piece(OppColor,king,OX1,OY1,X1,Y1,X2,Y2),
    square(OX2,OY2), |OX2-OX1|<=1, |OY2-OY1|<=1, (OX2,OY2)!=(OX1,OY1),
    not new_piece(OppColor,_,OX2,OY2,X1,Y1,X2,Y2).

% Check if opponent move removes check
opponent_escapes(X1,Y1,X2,Y2) :- 
    opponent_move(OppColor,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2),
    king(OppColor,KX,KY),
    side_to_move(Color),
    not still_in_check_after_escape(KX,KY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2).

still_in_check_after_escape(KX,KY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2) :- 
    opponent_move(_,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2),
    king(_,KX,KY),
    side_to_move(Color),
    final_piece(Color,_,AX,AY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2),
    final_attack_square(AX,AY,KX,KY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2).

final_piece(Color,Type,FX,FY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2) :- 
    opponent_move(_,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2),
    new_piece(Color,Type,FX,FY,X1,Y1,X2,Y2),
    (FX,FY)!=(OX1,OY1), (FX,FY)!=(OX2,OY2).

final_piece(OppColor,Type,OX2,OY2,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2) :- 
    opponent_move(OppColor,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2),
    new_piece(OppColor,Type,OX1,OY1,X1,Y1,X2,Y2).

final_attack_square(AX,AY,TX,TY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2) :- 
    final_piece(Color,pawn,AX,AY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2), Color=white,
    TX=AX+1, TY=AY+1, square(TX,TY).

final_attack_square(AX,AY,TX,TY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2) :- 
    final_piece(Color,pawn,AX,AY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2), Color=white,
    TX=AX-1, TY=AY+1, square(TX,TY).

final_attack_square(AX,AY,TX,TY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2) :- 
    final_piece(Color,pawn,AX,AY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2), Color=black,
    TX=AX+1, TY=AY-1, square(TX,TY).

final_attack_square(AX,AY,TX,TY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2) :- 
    final_piece(Color,pawn,AX,AY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2), Color=black,
    TX=AX-1, TY=AY-1, square(TX,TY).

final_attack_square(AX,AY,TX,TY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2) :- 
    final_piece(Color,rook,AX,AY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2),
    square(TX,TY), (TX=AX; TY=AY), (TX,TY)!=(AX,AY),
    not final_blocked_path(AX,AY,TX,TY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2).

final_attack_square(AX,AY,TX,TY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2) :- 
    final_piece(Color,bishop,AX,AY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2),
    square(TX,TY), |TX-AX|=|TY-AY|, (TX,TY)!=(AX,AY),
    not final_blocked_diagonal(AX,AY,TX,TY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2).

final_attack_square(AX,AY,TX,TY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2) :- 
    final_piece(Color,queen,AX,AY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2),
    square(TX,TY), ((TX=AX; TY=AY); |TX-AX|=|TY-AY|), (TX,TY)!=(AX,AY),
    not final_blocked_path(AX,AY,TX,TY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2),
    not final_blocked_diagonal(AX,AY,TX,TY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2).

final_attack_square(AX,AY,TX,TY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2) :- 
    final_piece(Color,knight,AX,AY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2),
    square(TX,TY),
    ((TX=AX+2,TY=AY+1); (TX=AX+2,TY=AY-1); (TX=AX-2,TY=AY+1); (TX=AX-2,TY=AY-1);
     (TX=AX+1,TY=AY+2); (TX=AX+1,TY=AY-2); (TX=AX-1,TY=AY+2); (TX=AX-1,TY=AY-2)).

final_attack_square(AX,AY,TX,TY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2) :- 
    final_piece(Color,king,AX,AY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2),
    square(TX,TY), |TX-AX|<=1, |TY-AY|<=1, (TX,TY)!=(AX,AY).

final_blocked_path(PX1,PY1,PX2,PY2,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2) :- 
    PX1=PX2, PY1<PY2,
    final_piece(_,_,PX1,PY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2), PY1<PY<PY2.

final_blocked_path(PX1,PY1,PX2,PY2,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2) :- 
    PX1=PX2, PY1>PY2,
    final_piece(_,_,PX1,PY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2), PY2<PY<PY1.

final_blocked_path(PX1,PY1,PX2,PY2,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2) :- 
    PY1=PY2, PX1<PX2,
    final_piece(_,_,PX,PY1,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2), PX1<PX<PX2.

final_blocked_path(PX1,PY1,PX2,PY2,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2) :- 
    PY1=PY2, PX1>PX2,
    final_piece(_,_,PX,PY1,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2), PX2<PX<PX1.

final_blocked_diagonal(PX1,PY1,PX2,PY2,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2) :- 
    PX1<PX2, PY1<PY2,
    final_piece(_,_,PX,PY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2), PX1<PX<PX2, PY1<PY<PY2, PX-PX1=PY-PY1.

final_blocked_diagonal(PX1,PY1,PX2,PY2,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2) :- 
    PX1<PX2, PY1>PY2,
    final_piece(_,_,PX,PY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2), PX1<PX<PX2, PY2<PY<PY1, PX-PX1=PY1-PY.

final_blocked_diagonal(PX1,PY1,PX2,PY2,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2) :- 
    PX1>PX2, PY1<PY2,
    final_piece(_,_,PX,PY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2), PX2<PX<PX1, PY1<PY<PY2, PX1-PX=PY-PY1.

final_blocked_diagonal(PX1,PY1,PX2,PY2,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2) :- 
    PX1>PX2, PY1>PY2,
    final_piece(_,_,PX,PY,OX1,OY1,OX2,OY2,X1,Y1,X2,Y2), PX2<PX<PX1, PY2<PY<PY1, PX1-PX=PY1-PY.

% Mate in one exists if opponent is in check and has no escape
mate_in_one(X1,Y1,X2,Y2) :- 
    opponent_in_check(X1,Y1,X2,Y2),
    not opponent_escapes(X1,Y1,X2,Y2).

#show mate_in_one/4.
```