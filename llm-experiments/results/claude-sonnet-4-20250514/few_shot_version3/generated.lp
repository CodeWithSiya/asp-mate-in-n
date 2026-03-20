```asp
% Board squares
square(1..8, 1..8).

% Generate candidate moves for the side to move
{ move(Color, Type, X1, Y1, X2, Y2) : square(X2, Y2) } :- 
    piece(Color, Type, X1, Y1), 
    side_to_move(Color).

% Legal move constraints for different piece types
:- move(Color, pawn, X1, Y1, X2, Y2), 
   Color = white,
   not ((X2 = X1, Y2 = Y1 + 1, not piece(_, _, X2, Y2)) ;
        (X2 = X1, Y2 = Y1 + 2, Y1 = 2, not piece(_, _, X2, Y2), not piece(_, _, X1, Y1 + 1)) ;
        (|X2 - X1| = 1, Y2 = Y1 + 1, piece(black, _, X2, Y2))).

:- move(Color, pawn, X1, Y1, X2, Y2), 
   Color = black,
   not ((X2 = X1, Y2 = Y1 - 1, not piece(_, _, X2, Y2)) ;
        (X2 = X1, Y2 = Y1 - 2, Y1 = 7, not piece(_, _, X2, Y2), not piece(_, _, X1, Y1 - 1)) ;
        (|X2 - X1| = 1, Y2 = Y1 - 1, piece(white, _, X2, Y2))).

:- move(Color, rook, X1, Y1, X2, Y2),
   not ((X1 = X2, Y1 != Y2, not blocked_vertical(X1, Y1, Y2)) ;
        (Y1 = Y2, X1 != X2, not blocked_horizontal(Y1, X1, X2))).

:- move(Color, bishop, X1, Y1, X2, Y2),
   not (|X2 - X1| = |Y2 - Y1|, X1 != X2, not blocked_diagonal(X1, Y1, X2, Y2)).

:- move(Color, queen, X1, Y1, X2, Y2),
   not ((X1 = X2, Y1 != Y2, not blocked_vertical(X1, Y1, Y2)) ;
        (Y1 = Y2, X1 != X2, not blocked_horizontal(Y1, X1, X2)) ;
        (|X2 - X1| = |Y2 - Y1|, X1 != X2, not blocked_diagonal(X1, Y1, X2, Y2))).

:- move(Color, knight, X1, Y1, X2, Y2),
   not ((|X2 - X1| = 2, |Y2 - Y1| = 1) ; (|X2 - X1| = 1, |Y2 - Y1| = 2)).

:- move(Color, king, X1, Y1, X2, Y2),
   not (|X2 - X1| <= 1, |Y2 - Y1| <= 1, (X1 != X2 ; Y1 != Y2)).

% Cannot move to square occupied by own piece
:- move(Color, Type, X1, Y1, X2, Y2), piece(Color, _, X2, Y2).

% Blocking predicates
blocked_vertical(X, Y1, Y2) :- Y1 < Y2, piece(_, _, X, Y), Y1 < Y < Y2.
blocked_vertical(X, Y1, Y2) :- Y1 > Y2, piece(_, _, X, Y), Y2 < Y < Y1.

blocked_horizontal(Y, X1, X2) :- X1 < X2, piece(_, _, X, Y), X1 < X < X2.
blocked_horizontal(Y, X1, X2) :- X1 > X2, piece(_, _, X, Y), X2 < X < X1.

blocked_diagonal(X1, Y1, X2, Y2) :- 
    X1 < X2, Y1 < Y2, piece(_, _, X, Y), 
    X1 < X < X2, Y1 < Y < Y2, X - X1 = Y - Y1.
blocked_diagonal(X1, Y1, X2, Y2) :- 
    X1 < X2, Y1 > Y2, piece(_, _, X, Y), 
    X1 < X < X2, Y2 < Y < Y1, X - X1 = Y1 - Y.
blocked_diagonal(X1, Y1, X2, Y2) :- 
    X1 > X2, Y1 < Y2, piece(_, _, X, Y), 
    X2 < X < X1, Y1 < Y < Y2, X1 - X = Y - Y1.
blocked_diagonal(X1, Y1, X2, Y2) :- 
    X1 > X2, Y1 > Y2, piece(_, _, X, Y), 
    X2 < X < X1, Y2 < Y < Y1, X1 - X = Y1 - Y.

% Position after move
piece_after(Color, Type, X2, Y2) :- move(Color, Type, X1, Y1, X2, Y2).
piece_after(Color, Type, X, Y) :- 
    piece(Color, Type, X, Y), 
    not move(Color, Type, X, Y, _, _),
    not captured(X, Y).

captured(X, Y) :- move(_, _, _, _, X, Y), piece(_, _, X, Y).

king_after(Color, X, Y) :- piece_after(Color, king, X, Y).

% Attack relations after move
attacks_after(Color, Type, X1, Y1, X2, Y2) :-
    piece_after(Color, Type, X1, Y1),
    square(X2, Y2),
    can_attack(Type, X1, Y1, X2, Y2).

can_attack(pawn, X1, Y1, X2, Y2) :- 
    piece_after(white, pawn, X1, Y1),
    |X2 - X1| = 1, Y2 = Y1 + 1.
can_attack(pawn, X1, Y1, X2, Y2) :- 
    piece_after(black, pawn, X1, Y1),
    |X2 - X1| = 1, Y2 = Y1 - 1.

can_attack(rook, X1, Y1, X2, Y2) :-
    ((X1 = X2, Y1 != Y2, not blocked_vertical_after(X1, Y1, Y2)) ;
     (Y1 = Y2, X1 != X2, not blocked_horizontal_after(Y1, X1, X2))).

can_attack(bishop, X1, Y1, X2, Y2) :-
    |X2 - X1| = |Y2 - Y1|, X1 != X2, 
    not blocked_diagonal_after(X1, Y1, X2, Y2).

can_attack(queen, X1, Y1, X2, Y2) :-
    ((X1 = X2, Y1 != Y2, not blocked_vertical_after(X1, Y1, Y2)) ;
     (Y1 = Y2, X1 != X2, not blocked_horizontal_after(Y1, X1, X2)) ;
     (|X2 - X1| = |Y2 - Y1|, X1 != X2, not blocked_diagonal_after(X1, Y1, X2, Y2))).

can_attack(knight, X1, Y1, X2, Y2) :-
    ((|X2 - X1| = 2, |Y2 - Y1| = 1) ; (|X2 - X1| = 1, |Y2 - Y1| = 2)).

can_attack(king, X1, Y1, X2, Y2) :-
    |X2 - X1| <= 1, |Y2 - Y1| <= 1, (X1 != X2 ; Y1 != Y2).

% Blocking predicates after move
blocked_vertical_after(X, Y1, Y2) :- Y1 < Y2, piece_after(_, _, X, Y), Y1 < Y < Y2.
blocked_vertical_after(X, Y1, Y2) :- Y1 > Y2, piece_after(_, _, X, Y), Y2 < Y < Y1.

blocked_horizontal_after(Y, X1, X2) :- X1 < X2, piece_after(_, _, X, Y), X1 < X < X2.
blocked_horizontal_after(Y, X1, X2) :- X1 > X2, piece_after(_, _, X, Y), X2 < X < X1.

blocked_diagonal_after(X1, Y1, X2, Y2) :- 
    X1 < X2, Y1 < Y2, piece_after(_, _, X, Y), 
    X1 < X < X2, Y1 < Y < Y2, X - X1 = Y - Y1.
blocked_diagonal_after(X1, Y1, X2, Y2) :- 
    X1 < X2, Y1 > Y2, piece_after(_, _, X, Y), 
    X1 < X < X2, Y2 < Y < Y1, X - X1 = Y1 - Y.
blocked_diagonal_after(X1, Y1, X2, Y2) :- 
    X1 > X2, Y1 < Y2, piece_after(_, _, X, Y), 
    X2 < X < X1, Y1 < Y < Y2, X1 - X = Y - Y1.
blocked_diagonal_after(X1, Y1, X2, Y2) :- 
    X1 > X2, Y1 > Y2, piece_after(_, _, X, Y), 
    X2 < X < X1, Y2 < Y < Y1, X1 - X = Y1 - Y.

% Opponent color
opponent(white, black).
opponent(black, white).

% Check after move
in_check_after(Color) :-
    king_after(Color, KX, KY),
    opponent(Color, OpColor),
    attacks_after(OpColor, _, _, _, KX, KY).

% Opponent has no legal escape moves
no_escape :-
    side_to_move(Color),
    opponent(Color, OpColor),
    not has_escape_move(OpColor).

has_escape_move(OpColor) :-
    piece_after(OpColor, Type, X1, Y1),
    square(X2, Y2),
    valid_escape_move(OpColor, Type, X1, Y1, X2, Y2).

valid_escape_move(OpColor, Type, X1, Y1, X2, Y2) :-
    legal_move_pattern(Type, X1, Y1, X2, Y2),
    not piece_after(OpColor, _, X2, Y2),
    not results_in_check(OpColor, Type, X1, Y1, X2, Y2).

legal_move_pattern(pawn, X1, Y1, X2, Y2) :-
    piece_after(white, pawn, X1, Y1),
    ((X2 = X1, Y2 = Y1 + 1, not piece_after(_, _, X2, Y2)) ;
     (X2 = X1, Y2 = Y1 + 2, Y1 = 2, not piece_after(_, _, X2, Y2), not piece_after(_, _, X1, Y1 + 1)) ;
     (|X2 - X1| = 1, Y2 = Y1 + 1, piece_after(black, _, X2, Y2))).

legal_move_pattern(pawn, X1, Y1, X2, Y2) :-
    piece_after(black, pawn, X1, Y1),
    ((X2 = X1, Y2 = Y1 - 1, not piece_after(_, _, X2, Y2)) ;
     (X2 = X1, Y2 = Y1 - 2, Y1 = 7, not piece_after(_, _, X2, Y2), not piece_after(_, _, X1, Y1 - 1)) ;
     (|X2 - X1| = 1, Y2 = Y1 - 1, piece_after(white, _, X2, Y2))).

legal_move_pattern(Type, X1, Y1, X2, Y2) :-
    Type != pawn,
    can_attack(Type, X1, Y1, X2, Y2).

results_in_check(OpColor, Type, X1, Y1, X2, Y2) :-
    % Simulate the opponent's move and check if their king is still in check
    king_after(OpColor, KX, KY),
    opponent(OpColor, Color),
    piece_after(Color, AType, AX, AY),
    can_attack_with_opp_move(AType, AX, AY, KX, KY, OpColor, Type, X1, Y1, X2, Y2).

can_attack_with_opp_move(AType, AX, AY, KX, KY, OpColor, MType, MX1, MY1, MX2, MY2) :-
    % King position after opponent move
    KXnew = KX, KYnew = KY,
    { KXnew = MX2, KYnew = MY2 } :- MType = king,
    % Attacker position (unchanged unless captured)
    not (AX = MX2, AY = MY2),
    % Check if attack is possible
    can_attack_simulation(AType, AX, AY, KXnew, KYnew, OpColor, MType, MX1, MY1, MX2, MY2).

can_attack_simulation(AType, AX, AY, KX, KY, OpColor, MType, MX1, MY1, MX2, MY2) :-
    can_attack(AType, AX, AY, KX, KY).

% Exactly one move leads to mate
1 { move(Color, Type, X1, Y1, X2, Y2) : piece(Color, Type, X1, Y1), square(X2, Y2) } 1 :-
    side_to_move(Color).

% Mate conditions
:- not in_check_after(OpColor), side_to_move(Color), opponent(Color, OpColor).
:- not no_escape.

mate_move(Color, Type, X1, Y1, X2, Y2) :- move(Color, Type, X1, Y1, X2, Y2).

#show mate_move/6.
```