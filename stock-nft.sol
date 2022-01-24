// contracts/GameItems.sol
// SPDX-License-identifier: MIT
pragma solidity ^0.6.0;

import "https://github.com/OpenZeppelin/openzeppelin-contracts/blob/v3.2.0/contracts/token/ERC1155/ERC1155.sol";

contract stock_nft is ERC1155 {

    string public company_name;
    address[] public board_members;
    mapping (address => bool) is_board_member;
    uint board_member_count;
    string public url;
    uint public origin_token_count;

    uint256 public token_id = 0;
    uint public proposal_count = 1;

    address public system_address;
    // uint public unique_token_issued;

    struct Founder {
        address founder_address;
        uint founder_token_id;
        string founding_date;
    }

    struct Proposal {
        address id;
        uint count;
        uint agreed_count;
    }

    Founder founder;
    Proposal proposal;

    mapping(uint => Proposal) public Proposals;
    mapping(address => bool) public signed;
    mapping(uint => mapping(address => bool)) public isConfirmed;

    constructor(string memory _company_name, address[] memory _board_members, 
        uint _token_count, string memory _founding_date, address _system_address) public  ERC1155("https://raw.githubusercontent.com/nc-collin/seed/master/{id}.json"){

        company_name = _company_name;
        board_members = _board_members;
        board_member_count = board_members.length;
        origin_token_count = _token_count;
        url = "https://raw.githubusercontent.com/nc-collin/seed/master/{id}.json";

        for (uint i=0; i < board_member_count; ++i) {
            is_board_member[board_members[i]] = true;
        }

        founder = Founder(msg.sender, token_id, _founding_date);

        system_address = _system_address;        
    }

    modifier isBoardMember() {
        require(is_board_member[msg.sender], "Not Board Member");
        _;
    }

    modifier founderPrivilege() {
        require(msg.sender == founder.founder_address);
        _;
    }

    function IssueProposal() external isBoardMember returns (uint) {
        Proposals[proposal_count] = Proposal(msg.sender, 0, 0);
        return proposal_count++;
    }

    function confirmProposal(uint proposal_id) external isBoardMember {
        require(!isConfirmed[proposal_id][msg.sender], "You already confirmed this proposal");
        isConfirmed[proposal_id][msg.sender] = true;
        Proposals[proposal_id].agreed_count += 1;
    }

    function stockIssue(uint proposal_id, uint _token_id) public founderPrivilege isBoardMember {
        require(Proposals[proposal_id].agreed_count == board_member_count);
        _mint(system_address, _token_id, origin_token_count, "");
        token_id++;
    }

    function stockReissue(uint proposal_id, uint _token_id, uint amount) public founderPrivilege isBoardMember {
        require(Proposals[proposal_id].agreed_count == board_member_count);
        _mint(system_address, _token_id, amount, "");
        token_id++;
    }

    function stockTransfer(address from, address to, uint _token_id, uint count) public founderPrivilege {
        safeTransferFrom(from, to, _token_id, count, "");
    }

    function redemption(address from, uint _token_id, uint count) public founderPrivilege {
        _burn(from, _token_id, count);
    }
}
