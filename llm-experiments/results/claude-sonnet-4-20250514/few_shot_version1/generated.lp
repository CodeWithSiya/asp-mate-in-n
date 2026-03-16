```asp
% Board coordinates
coord(1..8).

% Piece types
piece_type(pawn;rook;knight;bishop;queen;king).

% Colors
color(white;black).
opposite(white,black). opposite(black,white).

% Generate all possible moves for the side to move
{ move(Color,Type,X1,Y1,X2,Y2) : coord(X2), coord(Y2) } :- 
    piece(Color,Type,X1,Y1), side_to_move(Color).

% Pawn moves (simplified - forward one square)
:- move(Color,pawn,X1,Y1,X2,Y2), X2 != X1.
:- move(white,pawn,X1,Y1,X2,Y2), Y2 != Y1+1.
:- move(black,pawn,X1,Y1,X2,Y2), Y2 != Y1-1.

% Pawn captures (diagonal)
valid_pawn_capture(Color,X1,Y1,X2,Y2) :- 
    piece(Color,pawn,X1,Y1), piece(OpColor,_,X2,Y2),
    opposite(Color,OpColor), |X2-X1| = 1,
    Color = white, Y2 = Y1+1.
valid_pawn_capture(Color,X1,Y1,X2,Y2) :- 
    piece(Color,pawn,X1,Y1), piece(OpColor,_,X2,Y2),
    opposite(Color,OpColor), |X2-X1| = 1,
    Color = black, Y2 = Y1-1.

:- move(Color,pawn,X1,Y1,X2,Y2), X1 != X2, not valid_pawn_capture(Color,X1,Y1,X2,Y2).

% Rook moves (horizontal/vertical)
:- move(Color,rook,X1,Y1,X2,Y2), X1 != X2, Y1 != Y2.

% Knight moves (L-shape)
:- move(Color,knight,X1,Y1,X2,Y2), 
   not ((|X2-X1| = 2, |Y2-Y1| = 1) ; (|X2-X1| = 1, |Y2-Y1| = 2)).

% Bishop moves (diagonal)
:- move(Color,bishop,X1,Y1,X2,Y2), |X2-X1| != |Y2-Y1|.

% Queen moves (rook + bishop)
:- move(Color,queen,X1,Y1,X2,Y2), 
   not (X1 = X2 ; Y1 = Y2 ; |X2-X1| = |Y2-Y1|).

% King moves (one square in any direction)
:- move(Color,king,X1,Y1,X2,Y2), |X2-X1| > 1.
:- move(Color,king,X1,Y1,X2,Y2), |Y2-Y1| > 1.

% Cannot move to same square
:- move(Color,Type,X1,Y1,X1,Y1).

% Cannot capture own pieces
:- move(Color,Type,X1,Y1,X2,Y2), piece(Color,_,X2,Y2).

% Cannot move through pieces (simplified path checking)
blocked(X1,Y1,X2,Y2) :- 
    piece(_,_,X,Y), X1 < X, X < X2, Y = Y1, X1 != X2, Y1 = Y2.
blocked(X1,Y1,X2,Y2) :- 
    piece(_,_,X,Y), X2 < X, X < X1, Y = Y1, X1 != X2, Y1 = Y2.
blocked(X1,Y1,X2,Y2) :- 
    piece(_,_,X,Y), Y1 < Y, Y < Y2, X = X1, Y1 != Y2, X1 = X2.
blocked(X1,Y1,X2,Y2) :- 
    piece(_,_,X,Y), Y2 < Y, Y < Y1, X = X1, Y1 != Y2, X1 = X2.

:- move(Color,Type,X1,Y1,X2,Y2), Type != knight, blocked(X1,Y1,X2,Y2).

% Position after move
piece_after(Color,Type,X2,Y2) :- move(Color,Type,X1,Y1,X2,Y2).
piece_after(Color,Type,X,Y) :- 
    piece(Color,Type,X,Y), not move(Color,Type,X,Y,_,_),
    not piece(OpColor,_,X,Y) : move(OpColor,_,_,_,X,Y), opposite(Color,OpColor).

% King position after move
king_after(Color,X,Y) :- piece_after(Color,king,X,Y).

% Attack relation after move
attacks_after(Color,X1,Y1,X2,Y2) :- 
    piece_after(Color,Type,X1,Y1), coord(X2), coord(Y2),
    Type = pawn, opposite(Color,OpColor),
    Color = white, |X2-X1| = 1, Y2 = Y1+1.
attacks_after(Color,X1,Y1,X2,Y2) :- 
    piece_after(Color,Type,X1,Y1), coord(X2), coord(Y2),
    Type = pawn, opposite(Color,OpColor),
    Color = black, |X2-X1| = 1, Y2 = Y1-1.
attacks_after(Color,X1,Y1,X2,Y2) :- 
    piece_after(Color,rook,X1,Y1), coord(X2), coord(Y2),
    (X1 = X2 ; Y1 = Y2), not blocked_after(X1,Y1,X2,Y2).
attacks_after(Color,X1,Y1,X2,Y2) :- 
    piece_after(Color,bishop,X1,Y1), coord(X2), coord(Y2),
    |X2-X1| = |Y2-Y1|, not blocked_after(X1,Y1,X2,Y2).
attacks_after(Color,X1,Y1,X2,Y2) :- 
    piece_after(Color,queen,X1,Y1), coord(X2), coord(Y2),
    (X1 = X2 ; Y1 = Y2 ; |X2-X1| = |Y2-Y1|), not blocked_after(X1,Y1,X2,Y2).
attacks_after(Color,X1,Y1,X2,Y2) :- 
    piece_after(Color,knight,X1,Y1), coord(X2), coord(Y2),
    ((|X2-X1| = 2, |Y2-Y1| = 1) ; (|X2-X1| = 1, |Y2-Y1| = 2)).
attacks_after(Color,X1,Y1,X2,Y2) :- 
    piece_after(Color,king,X1,Y1), coord(X2), coord(Y2),
    |X2-X1| <= 1, |Y2-Y1| <= 1, (X1 != X2 ; Y1 != Y2).

blocked_after(X1,Y1,X2,Y2) :- 
    piece_after(_,_,X,Y), X1 < X, X < X2, Y = Y1, X1 != X2, Y1 = Y2.
blocked_after(X1,Y1,X2,Y2) :- 
    piece_after(_,_,X,Y), X2 < X, X < X1, Y = Y1, X1 != X2, Y1 = Y2.
blocked_after(X1,Y1,X2,Y2) :- 
    piece_after(_,_,X,Y), Y1 < Y, Y < Y2, X = X1, Y1 != Y2, X1 = X2.
blocked_after(X1,Y1,X2,Y2) :- 
    piece_after(_,_,X,Y), Y2 < Y, Y < Y1, X = X1, Y1 != Y2, X1 = X2.

% Check after move
in_check_after(OpColor) :- 
    side_to_move(Color), opposite(Color,OpColor),
    king_after(OpColor,KX,KY), attacks_after(Color,_,_,KX,KY).

% Opponent has no legal moves to escape check
opponent_move(OpColor,Type,X1,Y1,X2,Y2) :- 
    side_to_move(Color), opposite(Color,OpColor),
    piece_after(OpColor,Type,X1,Y1), coord(X2), coord(Y2),
    % Apply same movement rules for opponent
    Type = king, |X2-X1| <= 1, |Y2-Y1| <= 1, (X1 != X2 ; Y1 != Y2),
    not piece_after(OpColor,_,X2,Y2),
    not attacks_after(Color,_,_,X2,Y2).

has_escape :- opponent_move(_,_,_,_,_,_).

% Mate in one: opponent king in check and no escape moves
mate_in_one :- in_check_after(_), not has_escape.

% Only consider moves that result in mate
:- not mate_in_one.

#show move/6.
```